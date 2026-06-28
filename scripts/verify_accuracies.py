"""
verify_accuracies.py
Executes a fully transparent prediction and evaluation pipeline on the 126-patient cohort,
printing the exact sample counts, prediction matrices, and validation scores.
"""

import sys, os, joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             recall_score, precision_score, fbeta_score, auc, precision_recall_curve)

BASE = '/home/administrator/Desktop/RCC'

# Fix the LGBM custom loss function unpickling error
import __main__
def f2_weighted_loss(*args, **kwargs): pass
__main__.f2_weighted_loss = f2_weighted_loss

print("==================================================")
print("             VERIFYING ACCURACIES & SCORES        ")
print("==================================================")

# 1. Load clinical matrix to identify metastasis staging targets
clin_path = f'{BASE}/datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m'] == 'M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]

# Map clinical features exactly as done in notebooks/04_Late_Fusion_3Modality.py
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

# 2. Load genomic gene expression (54 genes)
M2_FEATURES = joblib.load(f'{BASE}/models/dataset_2/Model2_Features.pkl')
gen_path = f'{BASE}/datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]
available_genes = [g for g in M2_FEATURES if g in gen_df.columns]
gen_features = gen_df[available_genes]
for mg in [g for g in M2_FEATURES if g not in gen_df.columns]:
    gen_features[mg] = 0.0
gen_features = gen_features[M2_FEATURES]
gen_features = gen_features[~gen_features.index.duplicated(keep='first')]

# 3. Load Radiomics shape and texture features (49 features)
rad_features_list = joblib.load(f'{BASE}/models/dataset_3/Model3_Features.pkl')
rad_path = f'{BASE}/datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# 4. Aligned 126 patient cohort (Inner Join)
df = pd.concat([clin_df[['metastasis']], clin_features, gen_features, rad_df], axis=1, join='inner')
y_true = df['metastasis'].values
X_clin = df[expected_clin_cols].values
X_gen = df[M2_FEATURES].values
X_rad = df[rad_features_list].values

print(f" cohort size: {len(y_true)} patients")
print(f" - Metastasis (M1) cases: {sum(y_true == 1)}")
print(f" - Non-metastasis (M0) cases: {sum(y_true == 0)}")
print("--------------------------------------------------")

# 5. Make predictions
# Model 1 Clinical
model1 = joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER.pkl')
P1_raw = model1.predict(X_clin, raw_score=True)
P1 = 1.0 / (1.0 + np.exp(-P1_raw))

# Model 2 Genomic
model2 = joblib.load(f'{BASE}/models/dataset_2/Model2_Genomic_TCGA.pkl')
scaler_gen = joblib.load(f'{BASE}/models/dataset_2/Model2_Scaler.pkl')
P2 = model2.predict_proba(scaler_gen.transform(X_gen))[:, 1]

# Model 3 Imaging
model3_booster = xgb.Booster()
model3_booster.load_model(f'{BASE}/models/dataset_3/Model3_Imaging_TCGA.json')
scaler_rad = joblib.load(f'{BASE}/models/dataset_3/Model3_Scaler.pkl')
dmat = xgb.DMatrix(scaler_rad.transform(X_rad), feature_names=rad_features_list)
P3 = model3_booster.predict(dmat)

# Fusions
P_bef = []
prior = 18/126
log_prior_odds = np.log(prior / (1 - prior))
for i in range(len(y_true)):
    accum = log_prior_odds
    for p in [P1[i], P2[i], P3[i]]:
        p_c = np.clip(p, 1e-6, 1 - 1e-6)
        accum += np.log(p_c / (1 - p_c)) - log_prior_odds
    P_bef.append(1 / (1 + np.exp(-accum)))
P_bef = np.array(P_bef)

# Dempster-Shafer
def run_dst(p1, p2, p3):
    r1, r2, r3 = 2*abs(0.736-0.5), 2*abs(0.897-0.5), 2*abs(0.928-0.5)
    def mass(p, r): return {'M1': p*r, 'M0': (1-p)*r, 'U': 1-r}
    def combine(m1, m2):
        K = m1['M1']*m2['M0'] + m1['M0']*m2['M1']
        norm = 1 - K if K < 1.0 else 1e-6
        return {
            'M1': (m1['M1']*m2['M1'] + m1['M1']*m2['U'] + m1['U']*m2['M1']) / norm,
            'M0': (m1['M0']*m2['M0'] + m1['M0']*m2['U'] + m1['U']*m2['M0']) / norm,
            'U': (m1['U']*m2['U']) / norm
        }
    probs = []
    for i in range(len(y_true)):
        m1 = mass(np.clip(p1[i], 1e-6, 1-1e-6), r1)
        m2 = mass(np.clip(p2[i], 1e-6, 1-1e-6), r2)
        m3 = mass(np.clip(p3[i], 1e-6, 1-1e-6), r3)
        c = combine(m1, m2)
        c = combine(c, m3)
        probs.append(c['M1'] + 0.5 * c['U'])
    return np.array(probs)
P_dst = run_dst(P1, P2, P3)

# Optimal Transport
w1, w2, w3 = 0.1016, 0.3091, 0.5892
P_ot = []
for i in range(len(y_true)):
    num = w1*np.log(P1[i]/(1-P1[i])) + w2*np.log(P2[i]/(1-P2[i])) + w3*np.log(P3[i]/(1-P3[i]))
    den = w1 + w2 + w3 + 0.001
    P_ot.append(1 / (1 + np.exp(-num/den)))
P_ot = np.array(P_ot)

# 6. Evaluate all and print
def get_metrics(name, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec_c, prec_c)
    
    # 1. Standard Threshold (0.50)
    y_pred_50 = (y_prob >= 0.5).astype(int)
    f2_50 = fbeta_score(y_true, y_pred_50, beta=2, zero_division=0)
    rec_50 = recall_score(y_true, y_pred_50, zero_division=0)
    prec_50 = precision_score(y_true, y_pred_50, zero_division=0)
    
    # 2. Optimal F2-maximizing threshold targeting high recall >= 80%
    best_f2 = 0
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 200):
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= 0.80:
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_t = t
                
    y_pred_opt = (y_prob >= best_t).astype(int)
    f2_opt = fbeta_score(y_true, y_pred_opt, beta=2, zero_division=0)
    rec_opt = recall_score(y_true, y_pred_opt, zero_division=0)
    prec_opt = precision_score(y_true, y_pred_opt, zero_division=0)
    
    print(f"\nModel: {name}")
    print(f"  AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    print(f"  [Standard t=0.50]  Recall: {rec_50:.4f} | Precision: {prec_50:.4f} | F2-Score: {f2_50:.4f}")
    print(f"  [Optimised t={best_t:.4f}] Recall: {rec_opt:.4f} | Precision: {prec_opt:.4f} | F2-Score: {f2_opt:.4f}")

get_metrics("Model 1 (Clinical)", P1)
get_metrics("Model 2 (Genomic)", P2)
get_metrics("Model 3 (Imaging)", P3)
get_metrics("BEF (Bayesian Evidence Fusion)", P_bef)
get_metrics("DST (Dempster-Shafer Fusion)", P_dst)
get_metrics("OT (Optimal Transport Fusion)", P_ot)
print("==================================================")
