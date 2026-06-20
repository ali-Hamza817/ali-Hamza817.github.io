<div align="center">
  <img src="https://img.shields.io/badge/status-Thesis_Complete-gold?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/framework-Flask-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/AI-LightGBM_%7C_XGBoost_%7C_LinearSVC-orange?style=for-the-badge" alt="AI"/>
</div>

<h1 align="center">🔬 RCC·AI: Multi-Cohort Late Fusion Metastasis Predictor</h1>
<p align="center"><b>Decision-Level Multimodal Framework for Renal Cell Carcinoma</b></p>

---

## 🌟 What This Project Is

**RCC·AI** is a **decision-level late fusion** clinical decision support system that predicts distant metastasis in Renal Cell Carcinoma (RCC) patients. Three independently trained, single-modality models each produce a probability score; these scores are then aggregated via weighted averaging or a meta-learner to form a final prediction.

The three modalities:
- 🏥 **Clinical** — LightGBM trained on 36,738 SEER patients, applied to TCGA via transfer
- 🧬 **Genomic** — LinearSVC on a 54-gene ANOVA-selected RNA-Seq signature (418 TCGA-KIRC patients)
- 🫁 **Radiomic** — XGBoost on 49 PyRadiomics features from 3D CT scans (126 TCGA-KIRC patients)

### What This Project Is Not

This is **not** an end-to-end multimodal deep learning system and does **not** perform joint representation learning or a unified biological embedding. It is correctly described as **score-level (late) fusion** — the standard and scientifically appropriate methodology for this data setting (small N, heterogeneous sources, three independent cohorts).

---

## 🏆 Scientific Contributions

1. **Multi-cohort RCC pipeline:** Spans SEER (clinical, n=36,738) → TCGA-KIRC (genomic, n=418) → TCGA-KIRC imaging (n=126), connecting population-scale and patient-scale data.
2. **F2-optimised screening design:** Custom `f2_weighted_loss` (FN penalty 4×) combined with F2-threshold optimisation — mathematically aligns the model with the clinical priority of never missing metastasis.
3. **54-gene transcriptomic profile:** SelectKBest(ANOVA, k=50) + 5 literature genes from the full 19,000+ gene HiSeqV2 matrix, achieving 94.4% Recall on OOF evaluation.
4. **Full radiomics pipeline:** TotalSegmentator auto-segmentation → PyRadiomics 3D feature extraction across 126 TCGA-KIRC patients.
5. **Four fusion strategies compared:** Simple Average, F2-Weighted Average, Stacking Meta-Learner, Cascade Max Pooling — all evaluated on the same alignment cohort with no data leakage.
6. **Transparent limitations:** The fusion cohort is an inner-join alignment subset (n=126), not a natural multimodal dataset. All metrics are reported with this context.

---

## ⚠️ Important Methodological Notes

### Fusion Cohort
The 3-modality fusion was evaluated on a **harmonised inner-join alignment cohort** of exactly 126 patients who had simultaneously valid Clinical transfer features, RNA-Seq OOF predictions, and extracted PyRadiomics features. This is a constructed intersection, not a naturally occurring multimodal dataset. Results should be interpreted as proof-of-concept performance on this specific alignment subset.

### Fusion Type
This is **score-level (late) fusion** — NOT joint representation learning or end-to-end multimodal deep learning. Each model is trained independently and only their output probabilities are combined.

### Site-Specific Sub-Model Outputs
The four organ-specific sub-models (Lung, Bone, Liver, Brain) are calibrated for maximum sensitivity via F2-loss. Their raw sigmoid outputs are **relative risk indices**, not calibrated absolute probabilities. Precision is 1–8% at near-maximal Recall — this is the intended screening behaviour, not a model failure.

---

## 📊 Results

All metrics sourced from saved result CSV files. No values were synthesized.

### Individual Models

| Model | Cohort | n | AUROC | Recall | F2 |
|:---|:---|:---:|:---:|:---:|:---:|
| Model 1: Clinical — Overall Met | SEER Holdout | ~7,348 | **0.7704** | 62.07% | 0.3779 |
| Model 2: Genomic (54-gene LinearSVC) | TCGA-418 OOF | 418 | 0.6420 | 92.86% | 0.5242 |
| Model 3: Imaging (XGBoost Radiomics) | TCGA-126 OOF | 126 | 0.6591 | 100.0% | 0.5128 |

### 3-Modality Fusion (TCGA-126 Alignment Cohort — Primary Thesis Result)

| Strategy | AUROC | AUPRC | F1 | F2 | Recall | Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fusion A: Simple Average | 0.7927 | **0.4457** | 0.4000 | **0.5970** | 88.89% | 25.81% |
| **Fusion B: F2-Weighted** ⭐ | **0.7973** | **0.4457** | 0.3951 | 0.5926 | 88.89% | 25.40% |
| Fusion C: Stacking Meta-Learner | 0.7665 | 0.4356 | **0.4242** | 0.5833 | 77.78% | **29.17%** |
| Fusion D: Cascade Max Pooling | 0.7377 | 0.3824 | 0.4615 | 0.5660 | 66.67% | 35.29% |

⭐ Fusion B is the primary reported result (highest AUROC). The AUROC improvement from the best single model (0.7704) to best fusion (0.7973) is **+0.027** — a real but modest gain consistent with late fusion on a small alignment cohort.

---

## 🎯 Clinical Justification: Why F2 / Recall

Missing a distant metastasis (False Negative) is a potentially fatal clinical error. A false positive triggers additional imaging — an inconvenience, not harm. The F2 Score explicitly penalises False Negatives more than False Positives. At 88.9% Recall, ~1 in 4 flagged patients is a true M1 case — appropriate for a population-level screening tool that triggers further workup, not a definitive diagnosis.

---

## 📂 Architecture

### Base Models
- 🏥 **Model 1 (Clinical):** LightGBM · F2-weighted loss · Optuna HPO · SMOTE · SEER n=36,738
- 🧬 **Model 2 (Genomic):** CalibratedClassifierCV(LinearSVC, C=0.01) · 54 ANOVA genes · SMOTE · TCGA n=418
- 🫁 **Model 3 (Imaging):** XGBoost · 49 PyRadiomics features · StandardScaler · SMOTE · TCGA n=126

### Score-Level Late Fusion (n=126 alignment cohort)
- **Fusion A:** Simple arithmetic mean of three probabilities
- **Fusion B:** Weighted average where weights = individual model F2 scores
- **Fusion C:** LogisticRegression meta-learner (class_weight='balanced') via 5-Fold nested CV
- **Fusion D:** Cascade max pooling — `max(P1, P2, P3)` — the most sensitive trigger

---

## 🚀 Running the Web Application

```bash
pip install flask flask-cors xgboost lightgbm scikit-learn imbalanced-learn pandas numpy joblib pyradiomics
cd webapp
python3 app.py
# Open http://127.0.0.1:5050
```

The frontend (Vercel-hosted) supports manual entry, CSV upload, and live NIfTI CT scan extraction via PyRadiomics. The backend (Flask) runs all models and returns structured JSON predictions.

*(Datasets excluded via .gitignore due to file size. Trained model weights in `models/` directory are included.)*

---

<div align="center">
  <i>Masters Thesis · 2026</i><br/>
  <i>"A multi-cohort, decision-level late fusion system for RCC metastasis screening — AUROC 0.797 on 126-patient alignment cohort, 88.9% sensitivity."</i>
</div>
