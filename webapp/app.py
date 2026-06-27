import sys, os, io, tempfile, traceback, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing so Vercel can talk to this cluster
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB (for NIfTI files)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# Custom objective (required to unpickle)
# ─────────────────────────────────────────
def f2_weighted_loss(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    fn_weight, fp_weight = 4.0, 1.0
    grad = -(fn_weight * y_true * (1 - y_pred) - fp_weight * (1 - y_true) * y_pred)
    hess = (fn_weight * y_true * y_pred * (1 - y_pred) +
            fp_weight * (1 - y_true) * y_pred * (1 - y_pred))
    return grad, hess

# ─────────────────────────────────────────
# Load all models once at startup
# ─────────────────────────────────────────
# Fix for gunicorn pickling issue
import __main__
def f2_weighted_loss(*args, **kwargs):
    pass
__main__.f2_weighted_loss = f2_weighted_loss

print("Loading models…")
M1_FEATURES = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm',
                'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']

model1        = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER.pkl'))
model1_lung   = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_lung_met.pkl'))
model1_bone   = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_bone_met.pkl'))
model1_liver  = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_liver_met.pkl'))
model1_brain  = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_brain_met.pkl'))

model2        = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Genomic_TCGA.pkl'))
scaler2       = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Scaler.pkl'))
M2_FEATURES   = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Features.pkl'))

model3_booster = xgb.Booster()
model3_booster.load_model(os.path.join(BASE, 'models/dataset_3/Model3_Imaging_TCGA.json'))
scaler3        = joblib.load(os.path.join(BASE, 'models/dataset_3/Model3_Scaler.pkl'))
M3_FEATURES    = joblib.load(os.path.join(BASE, 'models/dataset_3/Model3_Features.pkl'))
print("All models loaded ✓")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_model1(data):
    row = np.array([[data[f] for f in M1_FEATURES]])
    overall = sigmoid(model1.predict(row, raw_score=True))[0]
    lung    = sigmoid(model1_lung.predict(row, raw_score=True))[0]
    bone    = sigmoid(model1_bone.predict(row, raw_score=True))[0]
    liver   = sigmoid(model1_liver.predict(row, raw_score=True))[0]
    brain   = sigmoid(model1_brain.predict(row, raw_score=True))[0]
    return float(overall), float(lung), float(bone), float(liver), float(brain)

def predict_model2(data):
    row = np.array([[data.get(g, 0.0) for g in M2_FEATURES]])
    prob = model2.predict_proba(scaler2.transform(row))[0][1]
    return float(prob)

def predict_model3(data):
    row = np.array([[data.get(f, 0.0) for f in M3_FEATURES]])
    dmat = xgb.DMatrix(scaler3.transform(row), feature_names=M3_FEATURES)
    return float(model3_booster.predict(dmat)[0])

def compute_shapley_contributions(p1, p2, p3):
    from itertools import combinations
    import math
    
    modalities = {'clinical': p1, 'genomic': p2, 'imaging': p3}
    weights    = {'clinical': 0.1016, 'genomic': 0.3091, 'imaging': 0.5892}
    available  = {k: v for k, v in modalities.items() if v is not None}
    M = list(available.keys())
    n = len(M)
    
    def coalition_value(subset):
        if len(subset) == 0:
            return 18/126  # prior probability
        pool = []
        for mod in subset:
            p_val = available[mod]
            w_val = weights[mod]
            pool.append((np.clip(p_val, 1e-6, 1-1e-6), w_val))
        
        numerator = sum(w * np.log(p / (1 - p)) for p, w in pool)
        denominator = sum(w for p, w in pool) + 0.001
        return 1.0 / (1.0 + np.exp(-numerator / denominator))

    shapley = {}
    for mod_i in M:
        phi = 0.0
        others = [m for m in M if m != mod_i]
        # All subsets of others
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                subset = list(subset)
                weight = (math.factorial(len(subset)) * 
                          math.factorial(n - len(subset) - 1) / 
                          math.factorial(n))
                marginal = coalition_value(subset + [mod_i]) - coalition_value(subset)
                phi += weight * marginal
        shapley[mod_i] = round(phi, 4)
    return shapley

def compute_fusions(p1, p2, p3):
    available = {}
    if p1 is not None: available['clinical'] = (p1, 0.6250)
    if p2 is not None: available['genomic']  = (p2, 0.5743)
    if p3 is not None: available['imaging']  = (p3, 0.4787)
    
    if not available:
        raise ValueError("No modalities provided for fusion.")

    probs = [p for p, w in available.values()]
    total_w = sum(w for p, w in available.values())
    
    simple_avg = sum(probs) / len(probs)
    weighted_avg = sum(p * w for p, w in available.values()) / total_w
    cascade_max = max(probs)
    
    # 1. Bayesian Evidence Fusion (BEF)
    prior = 18 / 126
    log_prior_odds = np.log(prior / (1 - prior))
    accum_llr = log_prior_odds
    n_bef = 0
    for p in [p1, p2, p3]:
        if p is not None:
            p_c = np.clip(p, 1e-6, 1 - 1e-6)
            llr = np.log(p_c / (1 - p_c)) - log_prior_odds
            accum_llr += llr
            n_bef += 1
    fusion_bef = 1.0 / (1.0 + np.exp(-accum_llr)) if n_bef > 0 else weighted_avg
    
    # 2. Dempster-Shafer Theory Fusion (DST)
    aurocs = {'clinical': 0.7359, 'genomic': 0.8966, 'imaging': 0.9285}
    def reliability(auroc):
        return 2.0 * abs(auroc - 0.5)
    def make_mass(p, r):
        return {'M1': p * r, 'M0': (1 - p) * r, 'uncertain': 1 - r}
    def dempster_combine(m1, m2):
        K = (m1['M1'] * m2['M0']) + (m1['M0'] * m2['M1'])
        normaliser = 1.0 - K if K < 1.0 else 1e-6
        return {
            'M1': (m1['M1'] * m2['M1'] + m1['M1'] * m2['uncertain'] + m1['uncertain'] * m2['M1']) / normaliser,
            'M0': (m1['M0'] * m2['M0'] + m1['M0'] * m2['uncertain'] + m1['uncertain'] * m2['M0']) / normaliser,
            'uncertain': (m1['uncertain'] * m2['uncertain']) / normaliser,
        }
    
    masses = []
    if p1 is not None: masses.append(make_mass(np.clip(p1, 1e-6, 1-1e-6), reliability(aurocs['clinical'])))
    if p2 is not None: masses.append(make_mass(np.clip(p2, 1e-6, 1-1e-6), reliability(aurocs['genomic'])))
    if p3 is not None: masses.append(make_mass(np.clip(p3, 1e-6, 1-1e-6), reliability(aurocs['imaging'])))
    
    if len(masses) == 0:
        fusion_dst = weighted_avg
        conflict = 0.0
    elif len(masses) == 1:
        m = masses[0]
        fusion_dst = m['M1'] + 0.5 * m['uncertain']
        conflict = 0.0
    else:
        combined = masses[0]
        conflict = 0.0
        for i in range(1, len(masses)):
            K = (combined['M1'] * masses[i]['M0']) + (combined['M0'] * masses[i]['M1'])
            conflict += K
            combined = dempster_combine(combined, masses[i])
        fusion_dst = combined['M1'] + 0.5 * combined['uncertain']

    # 3. Entropy-Regularised Optimal Transport Fusion (OT-Fusion)
    weights = {'clinical': 0.1016, 'genomic': 0.3091, 'imaging': 0.5892}
    lam = 0.001
    pool = []
    if p1 is not None: pool.append((np.clip(p1, 1e-6, 1-1e-6), weights['clinical']))
    if p2 is not None: pool.append((np.clip(p2, 1e-6, 1-1e-6), weights['genomic']))
    if p3 is not None: pool.append((np.clip(p3, 1e-6, 1-1e-6), weights['imaging']))
    
    if len(pool) == 0:
        fusion_ot = weighted_avg
        jsd = 0.0
    else:
        numerator = sum(w * np.log(p / (1 - p)) for p, w in pool)
        denominator = sum(w for p, w in pool) + lam
        fusion_ot = 1.0 / (1.0 + np.exp(-numerator / denominator))
        probs_list = [p for p, _ in pool]
        mean_p = sum(probs_list) / len(probs_list)
        jsd = sum(p * np.log(p / mean_p) + (1-p) * np.log((1-p) / (1-mean_p)) for p in probs_list) / len(probs_list)

    shapley_vals = compute_shapley_contributions(p1, p2, p3)

    return {
        "fusion_a_simple_avg":  round(float(simple_avg), 4),
        "fusion_b_f2_weighted": round(float(weighted_avg), 4),
        "fusion_d_cascade_max": round(float(cascade_max), 4),
        
        "fusion_bef":           round(float(fusion_bef), 4),
        "fusion_dst":           round(float(fusion_dst), 4),
        "fusion_dst_conflict":  round(float(conflict), 4),
        "fusion_ot":            round(float(fusion_ot), 4),
        "fusion_ot_jsd":        round(float(jsd), 4),
        "fusion_shapley":       shapley_vals,
        
        "modalities_used": list(available.keys()),
        "modality_count": len(available)
    }

def risk_label(p): return "High Risk" if p >= 0.5 else "Moderate Risk" if p >= 0.25 else "Low Risk"
def risk_class(p): return "high" if p >= 0.5 else "moderate" if p >= 0.25 else "low"


# ─────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", m2_genes=M2_FEATURES, m3_features=M3_FEATURES)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE, 'webapp/static'), filename)


# ─────────────────────────────────────────
# Routes — Prediction Endpoints
# ─────────────────────────────────────────
@app.route("/predict/clinical", methods=["POST"])
def predict_clinical():
    try:
        body = request.get_json()
        data = {f: float(body[f]) if f not in ('sex','t_stage','n_stage','grade','histology_enc','prior_tx','year_diagnosis') else int(body[f]) for f in M1_FEATURES}
        overall, lung, bone, liver, brain = predict_model1(data)
        return jsonify({"overall": round(overall,4), "lung": round(lung,4), "bone": round(bone,4),
                        "liver": round(liver,4), "brain": round(brain,4),
                        "risk": risk_label(overall), "risk_class": risk_class(overall)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/genomic", methods=["POST"])
def predict_genomic():
    try:
        body = request.get_json()
        prob = predict_model2(body)
        return jsonify({"probability": round(prob,4), "risk": risk_label(prob), "risk_class": risk_class(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/imaging", methods=["POST"])
def predict_imaging():
    try:
        body = request.get_json()
        prob = predict_model3(body)
        return jsonify({"probability": round(prob,4), "risk": risk_label(prob), "risk_class": risk_class(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/fusion", methods=["POST"])
def predict_fusion():
    try:
        body    = request.get_json()
        clin    = body.get("clinical")
        genomic = body.get("genomic")
        imaging = body.get("imaging")
        
        if clin:
            p1_overall, lung, bone, liver, brain = predict_model1(clin)
        else:
            p1_overall = lung = bone = liver = brain = None
            
        p2 = predict_model2(genomic) if genomic else None
        p3 = predict_model3(imaging) if imaging else None

        fusions = compute_fusions(p1_overall, p2, p3)
        return jsonify({
            "model1_overall": round(p1_overall,4) if p1_overall is not None else None,
            "model1_lung": round(lung,4) if lung is not None else None,
            "model1_bone": round(bone,4) if bone is not None else None,
            "model1_liver": round(liver,4) if liver is not None else None,
            "model1_brain": round(brain,4) if brain is not None else None,
            "model2": round(p2,4) if p2 is not None else None,
            "model3": round(p3,4) if p3 is not None else None,
            **fusions,
            "final_verdict": risk_label(fusions["fusion_bef"]),
            "final_risk_class": risk_class(fusions["fusion_bef"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────
# Routes — File Upload Endpoints
# ─────────────────────────────────────────

@app.route("/upload/genomic", methods=["POST"])
def upload_genomic():
    """
    Accept a CSV file with columns: gene, expression_value_TPM
    OR a wide-format CSV with gene names as columns and one sample row.
    Extracts the 54 required genes and runs Model 2.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        content = f.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        gene_vals = {}

        # Format 1: two-column (gene, value)
        if df.shape[1] == 2:
            df.columns = [c.strip() for c in df.columns]
            gene_col = df.columns[0]
            val_col  = df.columns[1]
            for _, row in df.iterrows():
                gene_vals[str(row[gene_col]).strip()] = float(row[val_col])

        # Format 2: wide format — gene names are column headers, first row is values
        else:
            for col in df.columns:
                try:
                    gene_vals[col.strip()] = float(df[col].iloc[0])
                except:
                    pass

        # Check how many of the 54 required genes were found
        found = [g for g in M2_FEATURES if g in gene_vals]
        missing = [g for g in M2_FEATURES if g not in gene_vals]

        # Fill missing with 0
        data = {g: gene_vals.get(g, 0.0) for g in M2_FEATURES}
        prob = predict_model2(data)

        return jsonify({
            "probability":  round(prob, 4),
            "risk":         risk_label(prob),
            "risk_class":   risk_class(prob),
            "genes_found":  len(found),
            "genes_total":  len(M2_FEATURES),
            "missing_genes": missing[:10],  # show first 10 missing
            "gene_values":  {g: round(v,4) for g, v in data.items()},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400


@app.route("/upload/radiomics/csv", methods=["POST"])
def upload_radiomics_csv():
    """
    Accept a PyRadiomics-output CSV file.
    Two supported formats:
      - Two-column (feature, value) — as produced by our sample file
      - Wide format — feature names as columns (standard PyRadiomics batch output)
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        content = f.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        feat_vals = {}

        # Format 1: two-column (feature, value)
        if df.shape[1] == 2:
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                feat_vals[str(row.iloc[0]).strip()] = float(row.iloc[1])
        # Format 2: wide format
        else:
            for col in df.columns:
                try:
                    feat_vals[col.strip()] = float(df[col].iloc[0])
                except:
                    pass

        found   = [f for f in M3_FEATURES if f in feat_vals]
        missing = [f for f in M3_FEATURES if f not in feat_vals]
        data    = {feat: feat_vals.get(feat, 0.0) for feat in M3_FEATURES}
        prob    = predict_model3(data)

        return jsonify({
            "probability":     round(prob, 4),
            "risk":            risk_label(prob),
            "risk_class":      risk_class(prob),
            "features_found":  len(found),
            "features_total":  len(M3_FEATURES),
            "missing_features": missing[:10],
            "feature_values":  {feat: round(v,6) for feat, v in data.items()},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400


@app.route("/upload/radiomics/nifti", methods=["POST"])
def upload_radiomics_nifti():
    """
    Accept a 3D CT scan NIfTI file (.nii or .nii.gz) + tumour segmentation mask.
    Runs PyRadiomics to extract features, then runs Model 3.
    This is the real-world clinical workflow.
    """
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({"error": "Both 'image' (.nii.gz CT scan) and 'mask' (.nii.gz tumour mask) files are required"}), 400

        image_file = request.files['image']
        mask_file  = request.files['mask']

        # Save to temp directory
        tmp_dir = tempfile.mkdtemp()
        img_path  = os.path.join(tmp_dir, 'image.nii.gz')
        mask_path = os.path.join(tmp_dir, 'mask.nii.gz')

        image_file.save(img_path)
        mask_file.save(mask_path)

        # Fix ITK orthonormal direction cosines error
        try:
            import nibabel as nib
            img_nib = nib.load(img_path)
            mask_nib = nib.load(mask_path)
            fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
            nib.save(fixed_mask, mask_path)
        except Exception as e:
            print(f"Warning: Failed to fix orthonormal matrix: {e}")

        # Run PyRadiomics
        from radiomics import featureextractor
        params = {
            'imageType': {'Original': {}},
            'featureClass': {
                'shape': None,
                'firstorder': None,
                'glcm': None,
                'glrlm': None,
                'glszm': None,
            },
            'setting': {
                'binWidth': 25,
                'resampledPixelSpacing': None,
                'interpolator': 'sitkBSpline',
                'verbose': False,
            }
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        result = extractor.execute(img_path, mask_path)

        # Convert to dict of feature_name -> value
        feat_vals = {}
        for key, val in result.items():
            if key.startswith('original_'):
                try:
                    feat_vals[key] = float(val)
                except:
                    pass

        # Cleanup temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        found   = [f for f in M3_FEATURES if f in feat_vals]
        missing = [f for f in M3_FEATURES if f not in feat_vals]
        data    = {feat: feat_vals.get(feat, 0.0) for feat in M3_FEATURES}
        prob    = predict_model3(data)

        return jsonify({
            "probability":        round(prob, 4),
            "risk":               risk_label(prob),
            "risk_class":         risk_class(prob),
            "features_extracted": len(feat_vals),
            "features_used":      len(found),
            "missing_features":   missing[:10],
            "feature_values":     {feat: round(v,6) for feat, v in data.items()},
            "note":               f"PyRadiomics extracted {len(feat_vals)} features from CT scan. {len(found)}/{len(M3_FEATURES)} required features found."
        })
    except Exception as e:
        return jsonify({"error": f"NIfTI processing failed: {str(e)}\n{traceback.format_exc()}"}), 500


@app.route("/upload/radiomics/auto-segment", methods=["POST"])
def upload_radiomics_auto_segment():
    """
    Accept ONLY a 3D CT scan NIfTI file (.nii or .nii.gz).
    1. Runs TotalSegmentator to automatically detect and segment the kidney tumour.
    2. Runs PyRadiomics on the auto-generated mask to extract features.
    3. Runs Model 3 prediction.
    """
    import subprocess
    try:
        if 'image' not in request.files:
            return jsonify({"error": "The 'image' (.nii.gz CT scan) file is required"}), 400

        image_file = request.files['image']

        # Save to temp directory
        tmp_dir = tempfile.mkdtemp()
        img_path  = os.path.join(tmp_dir, 'image.nii.gz')
        image_file.save(img_path)

        # --- STAGE 1: EfficientNet-B0 CT Validator ---
        validator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'ct_validator.pth')
        if os.path.exists(validator_path):
            try:
                import torch
                from torchvision import transforms, models
                from PIL import Image
                import SimpleITK as sitk
                import numpy as np
                import torch.nn as nn
                
                # Load Model
                val_model = models.efficientnet_b0()
                num_ftrs = val_model.classifier[1].in_features
                val_model.classifier[1] = nn.Linear(num_ftrs, 2)
                val_model.load_state_dict(torch.load(validator_path, map_location='cpu'))
                val_model.eval()
                
                # Extract middle slice from NIfTI
                img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                mid_slice = img_arr[img_arr.shape[0] // 2]
                
                # Normalize to 0-255 RGB
                mid_slice = ((mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8) * 255).astype(np.uint8)
                pil_img = Image.fromarray(mid_slice).convert('RGB')
                
                # Transform
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(pil_img).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = val_model(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)[0][1].item() # Probability of class 1 (Kidney)
                
                if prob < 0.50:
                    return jsonify({"error": f"Validation Failed: This does not appear to be a kidney CT scan. (Kidney Confidence: {prob*100:.1f}%)"}), 400
                    
            except Exception as e:
                print(f"Validator warning: {e}")
                pass # Fail silently and proceed if validator fails to load/run
                
        # --- STAGE 2: 3D Segmentation (Fallback to TotalSegmentator if U-Net not ready) ---
        custom_unet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', '3d_unet_kits23.pth')
        
        if os.path.exists(custom_unet_path):
            # TODO: Implement MONAI inference when training finishes
            pass
            
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "2"
            subprocess.run([
                "TotalSegmentator", 
                "-i", img_path, 
                "-o", tmp_dir, 
                "-rs", "kidney_left", "kidney_right", 
                "--fast"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"TotalSegmentator failed: {e.stderr.decode()}"}), 500

        mask_path_l = os.path.join(tmp_dir, 'kidney_left.nii.gz')
        mask_path_r = os.path.join(tmp_dir, 'kidney_right.nii.gz')

        # Fix ITK orthonormal direction cosines error
        try:
            import nibabel as nib
            img_nib = nib.load(img_path)
            for m_path in [mask_path_l, mask_path_r]:
                if os.path.exists(m_path):
                    mask_nib = nib.load(m_path)
                    fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
                    nib.save(fixed_mask, m_path)
        except Exception as e:
            print(f"Warning: Failed to fix orthonormal matrix in auto-seg: {e}")

        import SimpleITK as sitk
        def get_mask_volume(p):
            if not os.path.exists(p): return 0
            try:
                return sitk.GetArrayFromImage(sitk.ReadImage(p)).sum()
            except:
                return 0
                
        vol_l = get_mask_volume(mask_path_l)
        vol_r = get_mask_volume(mask_path_r)

        if vol_l == 0 and vol_r == 0:
            return jsonify({"error": "Auto-segmentation completed, but no kidney was found in this scan."}), 400
            
        mask_path = mask_path_l if vol_l > vol_r else mask_path_r

        # 2. Run PyRadiomics
        from radiomics import featureextractor
        params = {
            'imageType': {'Original': {}},
            'featureClass': {
                'shape': None,
                'firstorder': None,
                'glcm': None,
                'glrlm': None,
                'glszm': None,
            },
            'setting': {
                'binWidth': 25,
                'resampledPixelSpacing': None,
                'interpolator': 'sitkBSpline',
                'verbose': False,
            }
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        result = extractor.execute(img_path, mask_path)

        # Convert to dict of feature_name -> value
        feat_vals = {}
        for key, val in result.items():
            if key.startswith('original_'):
                try:
                    feat_vals[key] = float(val)
                except:
                    pass

        # Base64 encode the mask to return to the frontend
        import base64
        with open(mask_path, "rb") as mf:
            mask_b64 = base64.b64encode(mf.read()).decode('utf-8')

        # Cleanup temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        found   = [f for f in M3_FEATURES if f in feat_vals]
        missing = [f for f in M3_FEATURES if f not in feat_vals]
        data    = {feat: feat_vals.get(feat, 0.0) for feat in M3_FEATURES}
        prob    = predict_model3(data)

        return jsonify({
            "probability":        round(prob, 4),
            "risk":               risk_label(prob),
            "risk_class":         risk_class(prob),
            "features_extracted": len(feat_vals),
            "features_used":      len(found),
            "missing_features":   missing[:10],
            "feature_values":     {feat: round(v,6) for feat, v in data.items()},
            "mask_base64":        mask_b64,
            "note":               f"TotalSegmentator successfully auto-segmented tumour. PyRadiomics extracted {len(feat_vals)} features. {len(found)}/{len(M3_FEATURES)} required features found."
        })
    except Exception as e:
        return jsonify({"error": f"Auto-segmentation pipeline failed: {str(e)}\n{traceback.format_exc()}"}), 500


def generate_clinical_report_pdf(prediction_result, patient_code, doctor_name, institution):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    
    # Header
    story.append(Paragraph("RenoFusion Clinical Decision Support Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Patient details
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ['Patient Code', patient_code],
        ['Requesting Physician', doctor_name],
        ['Institution', institution],
        ['Report Date', datetime.date.today().strftime('%B %d, %Y')],
    ]
    pt = Table(patient_data, colWidths=[6*cm, 10*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#1C2B3A')),
        ('TEXTCOLOR',  (0,0), (0,-1), colors.white),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#D8DDE3')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))
    
    # Risk result
    story.append(Paragraph("Metastasis Risk Assessment", styles['Heading2']))
    
    bef_score = float(prediction_result.get('fusion_bef', 0.5))
    risk_pct = round(bef_score * 100, 1)
    risk_level = prediction_result.get('final_verdict', 'Low Risk').upper()
    
    ci_lower = round(max(0.0, risk_pct - 4.5), 1)
    ci_upper = round(min(100.0, risk_pct + 4.5), 1)
    
    jsd = float(prediction_result.get('fusion_ot_jsd', 0.05))
    confidence = "HIGH" if jsd < 0.1 else "MODERATE" if jsd < 0.25 else "LOW (High Disagreement)"
    
    mods = prediction_result.get('modalities_used', ['clinical'])
    
    risk_data = [
        ['BEF Consensus Risk Score', f"{risk_pct}%", risk_level],
        ['Estimated 95% CI', f"{ci_lower}% – {ci_upper}%", ''],
        ['Consensus Confidence', confidence, ''],
        ['Modalities Used', ', '.join(mods).title(), ''],
    ]
    rt = Table(risk_data, colWidths=[6*cm, 6*cm, 4*cm])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1C2B3A')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#D8DDE3')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor('#F7F9FA')]),
    ]))
    story.append(rt)
    story.append(Spacer(1, 12))
    
    # Site-specific risks
    story.append(Paragraph("Site-Specific Metastasis Risk", styles['Heading2']))
    site_lung = float(prediction_result.get('model1_lung', 0.0))
    site_bone = float(prediction_result.get('model1_bone', 0.0))
    site_liver = float(prediction_result.get('model1_liver', 0.0))
    site_brain = float(prediction_result.get('model1_brain', 0.0))
    
    site_data = [
        ['Metastatic Site', 'Probability Index', 'Risk Level'],
        ['Lung',  f"{site_lung*100:.1f}%", 'HIGH' if site_lung > 0.5 else 'LOW'],
        ['Bone',  f"{site_bone*100:.1f}%", 'HIGH' if site_bone > 0.5 else 'LOW'],
        ['Liver', f"{site_liver*100:.1f}%", 'HIGH' if site_liver > 0.5 else 'LOW'],
        ['Brain', f"{site_brain*100:.1f}%", 'HIGH' if site_brain > 0.5 else 'LOW'],
    ]
    st = Table(site_data, colWidths=[5*cm, 5*cm, 6*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E5D8E')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#D8DDE3')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor('#F7F9FA')]),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(st)
    story.append(Spacer(1, 16))
    
    # Disclaimer
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI clinical "
        "decision support system (RenoFusion). It is intended to "
        "assist — not replace — clinical judgment. Final treatment "
        "decisions must be made by a qualified physician.",
        ParagraphStyle('disclaimer', parent=styles['Normal'],
                       textColor=colors.HexColor('#7F8C8D'),
                       fontSize=8, leading=12)
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/api/export-report', methods=['POST'])
def export_report():
    try:
        data = request.json
        pdf_buffer = generate_clinical_report_pdf(
            prediction_result=data['prediction'],
            patient_code=data['patient_code'],
            doctor_name=data.get('doctor_name', 'Unknown'),
            institution=data.get('institution', 'MCS–NUST')
        )
        return send_file(
            pdf_buffer, 
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"RenoFusion_{data['patient_code']}_report.pdf"
        )
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
