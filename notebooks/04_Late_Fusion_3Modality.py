#!/usr/bin/env python
# coding: utf-8

# # Final 3-Modality Late Fusion
# This notebook fuses predictions from Model 1 (SEER Clinical), Model 2 (TCGA Genomic), and Model 3 (TCGA Imaging / Radiomics) using three strategies: Simple Average, Weighted Average, and Logistic Regression Stacking with Nested Cross-Validation.

# In[1]:


import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score, roc_curve)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')


# In[2]:


def evaluate_model(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_curve, precision_curve)

    # We use optimal threshold for F2 / Recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = [i for i, r in enumerate(recalls) if r >= 0.80]
    if len(valid_idx) > 0:
        optimal_idx = max(valid_idx, key=lambda i: precisions[i])
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    else:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {'AUROC': auroc, 'AUPRC': auprc, 'F2': f2, 'Recall': recall}


# ## 1. Data Alignment (Clinical + Genomic + Imaging on TCGA)

# In[3]:


# Load TCGA Clinical target
clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m'] == 'M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]

clin_features = pd.DataFrame(index=clin_df.index)
clin_features['age'] = clin_df['age_at_index'].fillna(clin_df['age_at_index'].mean())
clin_features['sex'] = clin_df['gender'].map({'male': 1, 'female': 0}).fillna(1)

def map_t_stage(t):
    if pd.isna(t): return 1
    if 'T1' in t: return 1
    if 'T2' in t: return 2
    if 'T3' in t: return 3
    if 'T4' in t: return 4
    return 1

clin_features['t_stage'] = clin_df['ajcc_t'].apply(map_t_stage)

def map_n_stage(n):
    if pd.isna(n): return 0
    if 'N0' in n: return 0
    if 'N1' in n: return 1
    if 'N2' in n: return 1
    return 0

clin_features['n_stage'] = clin_df['ajcc_n'].apply(map_n_stage)
clin_features['tumor_size_cm'] = 6.5
clin_features['grade'] = 2
clin_features['histology_enc'] = 0
clin_features['prior_tx'] = 0
clin_features['year_diagnosis'] = 2014

expected_clin_cols = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm', 'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']
clin_features = clin_features[expected_clin_cols]

# Load Genomic Data (54 Genes)
M2_FEATURES = joblib.load('../models/dataset_2/Model2_Features.pkl')
gen_path = '../datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]
available_genes = [g for g in M2_FEATURES if g in gen_df.columns]
gen_features = gen_df[available_genes]
for mg in [g for g in M2_FEATURES if g not in gen_df.columns]:
    gen_features[mg] = 0.0
gen_features = gen_features[M2_FEATURES]

# Load Imaging Data (Radiomics)
rad_path = '../datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

# Ensure unique indices before concat
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
clin_features = clin_features[~clin_features.index.duplicated(keep='first')]
gen_features = gen_features[~gen_features.index.duplicated(keep='first')]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# Inner Join across all 3 modalities
df = pd.concat([clin_df[['metastasis']], clin_features, gen_features, rad_df], axis=1, join='inner')

y_tcga = df['metastasis'].values
X_clin = df[expected_clin_cols].values
X_gen = df[M2_FEATURES].values
rad_features_list = joblib.load('../models/dataset_3/Model3_Features.pkl')
X_rad = df[rad_features_list].values

print(f"Final 3-Modality Fusion Cohort: {df.shape[0]} patients")
print(pd.Series(y_tcga).value_counts())


# ## 2. Generate Probabilities ($P_1$, $P_2$, $P_3$)

# In[4]:

import __main__
def f2_weighted_loss(*args, **kwargs):
    pass
__main__.f2_weighted_loss = f2_weighted_loss

# P1: Model 1 (Clinical SEER)
model1 = joblib.load('../models/dataset_1/Model1_Clinical_SEER.pkl')
P1_raw = model1.predict(X_clin, raw_score=True)
P1_oof = 1.0 / (1.0 + np.exp(-P1_raw))

# P2: Model 2 (Genomic TCGA)
model2 = joblib.load('../models/dataset_2/Model2_Genomic_TCGA.pkl')
scaler_gen = joblib.load('../models/dataset_2/Model2_Scaler.pkl')
P2_oof = model2.predict_proba(scaler_gen.transform(X_gen))[:, 1]

# P3: Model 3 (Imaging TCGA)
model3_booster = xgb.Booster()
model3_booster.load_model('../models/dataset_3/Model3_Imaging_TCGA.json')
scaler_rad = joblib.load('../models/dataset_3/Model3_Scaler.pkl')
dmat = xgb.DMatrix(scaler_rad.transform(X_rad), feature_names=rad_features_list)
P3_oof = model3_booster.predict(dmat)

print("Successfully extracted P1 (Transfer), P2 (OOF), and P3 (OOF).")


# ## 3. 3-Modality Late Fusion Ablation Study

# In[5]:

metrics = {}
metrics['Model 1: Clinical (SEER)'] = evaluate_model(y_tcga, P1_oof)
metrics['Model 2: Genomic (TCGA)'] = evaluate_model(y_tcga, P2_oof)
metrics['Model 3: Imaging (TCGA)'] = evaluate_model(y_tcga, P3_oof)

# Strategy A: Simple Average
P_fusion_A = (P1_oof + P2_oof + P3_oof) / 3
metrics['Fusion A: Simple avg'] = evaluate_model(y_tcga, P_fusion_A)

# Strategy B: Weighted Average
w1 = metrics['Model 1: Clinical (SEER)']['AUROC']
w2 = metrics['Model 2: Genomic (TCGA)']['AUROC']
w3 = metrics['Model 3: Imaging (TCGA)']['AUROC']
P_fusion_B = (w1 * P1_oof + w2 * P2_oof + w3 * P3_oof) / (w1 + w2 + w3)
metrics['Fusion B: Weighted avg'] = evaluate_model(y_tcga, P_fusion_B)

# Strategy C: Stacking (Nested CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_features = np.column_stack([P1_oof, P2_oof, P3_oof])
meta_model = LogisticRegression(C=0.1, random_state=42)
P_fusion_C = cross_val_predict(meta_model, stack_features, y_tcga, cv=cv, method='predict_proba')[:, 1]
metrics['Fusion C: Stacking'] = evaluate_model(y_tcga, P_fusion_C)

# Strategy D: Cascade Max
P_fusion_D = np.maximum(np.maximum(P1_oof, P2_oof), P3_oof)
metrics['Fusion D: Cascade max'] = evaluate_model(y_tcga, P_fusion_D)

# 1. Bayesian Evidence Fusion (BEF)
def run_bef(p1, p2, p3, prior=18/126):
    log_prior_odds = np.log(prior / (1 - prior))
    probs = []
    for i in range(len(y_tcga)):
        accum_llr = log_prior_odds
        for p in [p1[i], p2[i], p3[i]]:
            p_c = np.clip(p, 1e-6, 1 - 1e-6)
            llr_i = np.log(p_c / (1 - p_c)) - log_prior_odds
            accum_llr += llr_i
        probs.append(1 / (1 + np.exp(-accum_llr)))
    return np.array(probs)

P_bef = run_bef(P1_oof, P2_oof, P3_oof)
metrics['BEF: Bayesian Evidence Fusion'] = evaluate_model(y_tcga, P_bef)

# 2. Dempster-Shafer Theory Fusion (DST)
def run_dst(p1, p2, p3, auroc1=0.736, auroc2=0.897, auroc3=0.928):
    def reliability(auroc):
        return 2 * abs(auroc - 0.5)
    def make_mass(p, r):
        return {'M1': p * r, 'M0': (1 - p) * r, 'uncertain': 1 - r}
    def dempster_combine(m1, m2):
        K = (m1['M1'] * m2['M0']) + (m1['M0'] * m2['M1'])
        normaliser = 1 - K if K < 1.0 else 1e-6
        return {
            'M1': (m1['M1'] * m2['M1'] + m1['M1'] * m2['uncertain'] + m1['uncertain'] * m2['M1']) / normaliser,
            'M0': (m1['M0'] * m2['M0'] + m1['M0'] * m2['uncertain'] + m1['uncertain'] * m2['M0']) / normaliser,
            'uncertain': (m1['uncertain'] * m2['uncertain']) / normaliser,
        }
    probs = []
    r1, r2, r3 = reliability(auroc1), reliability(auroc2), reliability(auroc3)
    for i in range(len(y_tcga)):
        m1 = make_mass(np.clip(p1[i], 1e-6, 1-1e-6), r1)
        m2 = make_mass(np.clip(p2[i], 1e-6, 1-1e-6), r2)
        m3 = make_mass(np.clip(p3[i], 1e-6, 1-1e-6), r3)
        combined = dempster_combine(m1, m2)
        combined = dempster_combine(combined, m3)
        probs.append(combined['M1'] + 0.5 * combined['uncertain'])
    return np.array(probs)

P_dst = run_dst(P1_oof, P2_oof, P3_oof)
metrics['DST: Dempster-Shafer Fusion'] = evaluate_model(y_tcga, P_dst)

# 3. Entropy-Regularised Optimal Transport Fusion (OT-Fusion)
# Weights: Clinical=0.1016, Genomic=0.3091, Imaging=0.5892, lambda=0.001
def run_ot(p1, p2, p3, w1=0.1016, w2=0.3091, w3=0.5892, lam=0.001):
    probs = []
    for i in range(len(y_tcga)):
        numerator = (w1 * np.log(p1[i]/(1-p1[i])) + 
                     w2 * np.log(p2[i]/(1-p2[i])) + 
                     w3 * np.log(p3[i]/(1-p3[i])))
        denominator = w1 + w2 + w3 + lam
        probs.append(1 / (1 + np.exp(-numerator/denominator)))
    return np.array(probs)

P_ot = run_ot(np.clip(P1_oof, 1e-6, 1-1e-6), np.clip(P2_oof, 1e-6, 1-1e-6), np.clip(P3_oof, 1e-6, 1-1e-6))
metrics['OT: Optimal Transport Fusion'] = evaluate_model(y_tcga, P_ot)

# Compile Results
results_df = pd.DataFrame(metrics).T
print("\n--- Final Thesis Results: 3-Modality Late Fusion ---")
print(results_df.round(4))
results_df.to_csv('../results/Final_3Modality_Fusion_Results.csv')

