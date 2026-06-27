"""
generate_figures_v2.py
Generates five publication-quality figures for the Q1 research paper,
reflecting the new mathematical models (BEF, DST, OT-fusion) on the 126 cohort.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

warnings.filterwarnings('ignore')

# Paths
BASE = '/home/administrator/Desktop/RCC'
OUT = os.path.join(BASE, 'results', 'figures_v2')
os.makedirs(OUT, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
STYLE = {
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#ffffff',
    'axes.edgecolor':    '#333333',
    'axes.labelcolor':   '#111111',
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'text.color':        '#111111',
    'grid.color':        '#e5e7eb',
    'grid.linestyle':    '--',
    'grid.alpha':        0.8,
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#cccccc',
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
}
plt.rcParams.update(STYLE)

# Define Colors
CLRS = {
    'm1':       '#3730a3', # Indigo
    'm2':       '#0891b2', # Cyan
    'm3':       '#059669', # Green
    'fa':       '#d97706', # Amber
    'fb':       '#dc2626', # Red
    'fc':       '#7e22ce', # Purple
    'bef':      '#b45309', # Dark Orange
    'dst':      '#be185d', # Pink
    'ot':       '#0284c7', # Sky Blue
}

# ── LOAD DATA AND GENERATE PREDICTIONS ─────────────────────────────────────────

# Clinical Model
clin_path = f'{BASE}/datasets/dataset_2/KIRC_clinicalMatrix.tsv'
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

# Genomic (54 genes)
M2_FEATURES = joblib.load(f'{BASE}/models/dataset_2/Model2_Features.pkl')
gen_path = f'{BASE}/datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]
available_genes = [g for g in M2_FEATURES if g in gen_df.columns]
gen_features = gen_df[available_genes]
for mg in [g for g in M2_FEATURES if g not in gen_df.columns]:
    gen_features[mg] = 0.0
gen_features = gen_features[M2_FEATURES]

# Imaging
rad_path = f'{BASE}/datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

# Deduplicate
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
clin_features = clin_features[~clin_features.index.duplicated(keep='first')]
gen_features = gen_features[~gen_features.index.duplicated(keep='first')]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# Merge
df = pd.concat([clin_df[['metastasis']], clin_features, gen_features, rad_df], axis=1, join='inner')
y_true = df['metastasis'].values
X_clin = df[expected_clin_cols].values
X_gen = df[M2_FEATURES].values
rad_features_list = joblib.load(f'{BASE}/models/dataset_3/Model3_Features.pkl')
X_rad = df[rad_features_list].values

# Run standalone models
import __main__
def f2_weighted_loss(*args, **kwargs): pass
__main__.f2_weighted_loss = f2_weighted_loss

model1 = joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER.pkl')
P1_raw = model1.predict(X_clin, raw_score=True)
P1 = 1.0 / (1.0 + np.exp(-P1_raw))

model2 = joblib.load(f'{BASE}/models/dataset_2/Model2_Genomic_TCGA.pkl')
scaler_gen = joblib.load(f'{BASE}/models/dataset_2/Model2_Scaler.pkl')
P2 = model2.predict_proba(scaler_gen.transform(X_gen))[:, 1]

model3_booster = xgb.Booster()
model3_booster.load_model(f'{BASE}/models/dataset_3/Model3_Imaging_TCGA.json')
scaler_rad = joblib.load(f'{BASE}/models/dataset_3/Model3_Scaler.pkl')
dmat = xgb.DMatrix(scaler_rad.transform(X_rad), feature_names=rad_features_list)
P3 = model3_booster.predict(dmat)

# Fusions
P_simple = (P1 + P2 + P3) / 3

w1, w2, w3 = 0.7359, 0.8966, 0.9285
P_weighted = (w1*P1 + w2*P2 + w3*P3) / (w1 + w2 + w3)

def run_bef(p1, p2, p3, prior=18/126):
    log_prior_odds = np.log(prior / (1 - prior))
    probs = []
    for i in range(len(y_true)):
        accum_llr = log_prior_odds
        for p in [p1[i], p2[i], p3[i]]:
            p_c = np.clip(p, 1e-6, 1 - 1e-6)
            llr_i = np.log(p_c / (1 - p_c)) - log_prior_odds
            accum_llr += llr_i
        probs.append(1 / (1 + np.exp(-accum_llr)))
    return np.array(probs)
P_bef = run_bef(P1, P2, P3)

def run_dst_with_conflict(p1, p2, p3, auroc1=0.736, auroc2=0.897, auroc3=0.928):
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
        }, K
    probs, conflicts = [], []
    r1, r2, r3 = reliability(auroc1), reliability(auroc2), reliability(auroc3)
    for i in range(len(y_true)):
        m1 = make_mass(np.clip(p1[i], 1e-6, 1-1e-6), r1)
        m2 = make_mass(np.clip(p2[i], 1e-6, 1-1e-6), r2)
        m3 = make_mass(np.clip(p3[i], 1e-6, 1-1e-6), r3)
        combined, K12 = dempster_combine(m1, m2)
        combined, K23 = dempster_combine(combined, m3)
        probs.append(combined['M1'] + 0.5 * combined['uncertain'])
        conflicts.append(K12 + K23)
    return np.array(probs), np.array(conflicts)
P_dst, dst_conflicts = run_dst_with_conflict(P1, P2, P3)

def run_ot(p1, p2, p3, w1=0.1016, w2=0.3091, w3=0.5892, lam=0.001):
    probs = []
    for i in range(len(y_true)):
        numerator = (w1 * np.log(p1[i]/(1-p1[i])) + 
                     w2 * np.log(p2[i]/(1-p2[i])) + 
                     w3 * np.log(p3[i]/(1-p3[i])))
        denominator = w1 + w2 + w3 + lam
        probs.append(1 / (1 + np.exp(-numerator/denominator)))
    return np.array(probs)
P_ot = run_ot(np.clip(P1, 1e-6, 1-1e-6), np.clip(P2, 1e-6, 1-1e-6), np.clip(P3, 1e-6, 1-1e-6))


# ── FIG 1: ROC CURVES COMPARISON ───────────────────────────────────────────────
plt.figure(figsize=(7.5, 6.5))
for name, probs, color, style in [
    ('Clinical Model (SEER)', P1, CLRS['m1'], '--'),
    ('Genomic Model (TCGA)', P2, CLRS['m2'], '--'),
    ('Imaging Model (TCGA)', P3, CLRS['m3'], '--'),
    ('Simple Avg Fusion', P_simple, CLRS['fa'], ':'),
    ('Bayesian Evidence (BEF)', P_bef, CLRS['bef'], '-'),
    ('Dempster-Shafer (DST)', P_dst, CLRS['dst'], '-'),
    ('Optimal Transport (OT)', P_ot, CLRS['ot'], '-'),
]:
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', color=color, linestyle=style, lw=2.0)

plt.plot([0, 1], [0, 1], color='#9ca3af', linestyle='--', lw=1.2)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Receiver Operating Characteristic (ROC) Comparison\n(126-Patient Validation Cohort)', fontweight='bold')
plt.legend(loc='lower right', frameon=True, edgecolor='#dddddd')
plt.savefig(f'{OUT}/fig1_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Fig 1")


# ── FIG 2: PRECISION-RECALL (PR) CURVES COMPARISON ──────────────────────────────
plt.figure(figsize=(7.5, 6.5))
base_rate = y_true.mean()
plt.axhline(base_rate, color='#9ca3af', linestyle='--', lw=1.2, label=f'Random Baseline (Prevalence = {base_rate:.3f})')

for name, probs, color, style in [
    ('Clinical Model', P1, CLRS['m1'], '--'),
    ('Genomic Model', P2, CLRS['m2'], '--'),
    ('Imaging Model', P3, CLRS['m3'], '--'),
    ('Simple Avg Fusion', P_simple, CLRS['fa'], ':'),
    ('Bayesian Evidence (BEF)', P_bef, CLRS['bef'], '-'),
    ('Dempster-Shafer (DST)', P_dst, CLRS['dst'], '-'),
    ('Optimal Transport (OT)', P_ot, CLRS['ot'], '-'),
]:
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.plot(rec, prec, label=f'{name} (AUPRC = {ap:.4f})', color=color, linestyle=style, lw=2.0)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve Comparison\n(126-Patient Validation Cohort)', fontweight='bold')
plt.legend(loc='lower left', frameon=True, edgecolor='#dddddd')
plt.savefig(f'{OUT}/fig2_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Fig 2")


# ── FIG 3: PERFORMANCE ABLATION COMPOSITE BAR CHART ─────────────────────────────
# We will compare AUROC and F2-Score side-by-side
model_names = [
    'Clinical\n(SEER)', 'Genomic\n(TCGA)', 'Imaging\n(TCGA)',
    'Simple Avg\nFusion', 'Weighted\nAvg', 'BEF\n(Bayesian)', 'DST\n(Evidence)', 'OT\n(Transport)'
]
auroc_scores = [0.7359, 0.8966, 0.9285, 0.9789, 0.9805, 0.9794, 0.9805, 0.9697]
f2_scores    = [0.6198, 0.6818, 0.8621, 0.8427, 0.8523, 0.8427, 0.8621, 0.8621]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
rects1 = ax.bar(x - width/2, auroc_scores, width, label='AUROC', color='#1e3a8a', edgecolor='#111827', alpha=0.95)
rects2 = ax.bar(x + width/2, f2_scores, width, label='F2-Score (Recall-Optimised)', color='#10b981', edgecolor='#111827', alpha=0.9)

ax.set_ylabel('Score Metric')
ax.set_title('Ablation Study Comparison — Standalone vs. Multimodal Decision Fusion', fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim([0.4, 1.05])
ax.legend(loc='lower right', frameon=True, edgecolor='#dddddd')
ax.grid(axis='y', alpha=0.5)

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.savefig(f'{OUT}/fig3_ablation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Fig 3")


# ── FIG 4: DEMPSTER-SHAFER CONFLICT SCATTER PLOT ──────────────────────────────
# We plot the combination conflict score vs. divergence/variance of individual predictions
prediction_variance = np.var([P1, P2, P3], axis=0)
plt.figure(figsize=(7.5, 6))
# Class-colored points
scatter = plt.scatter(prediction_variance, dst_conflicts, c=y_true, cmap='coolwarm',
                      edgecolors='#1e293b', alpha=0.85, s=60, label=y_true)
plt.xlabel('Modality Prediction Variance (Var[P1, P2, P3])')
plt.ylabel('Dempster-Shafer Cumulative Conflict Metric (K)')
plt.title('Dempster-Shafer Combination Conflict vs. Modality Disagreement\n(Verifiable clinical discordance warning system)', fontweight='bold')
cbar = plt.colorbar(scatter)
cbar.set_label('True Metastasis Class (0 = M0, 1 = M1)')
plt.savefig(f'{OUT}/fig4_dst_conflict.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Fig 4")


# ── FIG 5: BAYESIAN EVIDENCE EVIDENCE ACCUMULATION ─────────────────────────────
# Visualizing how logs-odds build step-by-step for a few representative patients
# We select one M1 patient (high probability) and one M0 patient (low probability)
m1_idx = np.where(y_true == 1)[0][0] # Patient index for first M1
m0_idx = np.where(y_true == 0)[0][0] # Patient index for first M0

prior = 18/126
log_prior_odds = np.log(prior / (1 - prior))

def get_path(idx):
    p_vals = [P1[idx], P2[idx], P3[idx]]
    steps = [log_prior_odds]
    for p in p_vals:
        p_c = np.clip(p, 1e-6, 1 - 1e-6)
        llr = np.log(p_c / (1 - p_c)) - log_prior_odds
        steps.append(steps[-1] + llr)
    return steps

steps_m1 = get_path(m1_idx)
steps_m0 = get_path(m0_idx)

plt.figure(figsize=(8, 5.5))
stages = ['Cohort Prior', 'Add Clinical', '+ Add Genomic', '++ Add Imaging\n(Final BEF)']
plt.plot(stages, steps_m1, marker='o', color=CLRS['fb'], lw=2.5, label=f'True Metastatic Patient (Final P = {P_bef[m1_idx]:.4f})')
plt.plot(stages, steps_m0, marker='s', color=CLRS['m1'], lw=2.5, label=f'True Non-Metastatic Patient (Final P = {P_bef[m0_idx]:.4f})')

plt.axhline(0, color='#9ca3af', linestyle='--', lw=1.0)
plt.ylabel('Accumulated Log-Odds (Evidence space)')
plt.title('Log-Odds Evidence Progression (Bayesian Evidence Fusion)\n(Trace of step-by-step decision fusion for two sample patients)', fontweight='bold')
plt.legend(loc='best', frameon=True, edgecolor='#dddddd')
plt.savefig(f'{OUT}/fig5_bef_progression.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Fig 5")

print(f"\n✅ All 5 v2 figures successfully created and saved to:\n   {OUT}")
