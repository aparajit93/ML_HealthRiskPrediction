import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np

def save_model_and_threshold(model, threshold, model_path='xgb_best_model.joblib', threshold_path='threshold.txt'):
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save threshold
    with open(threshold_path, 'w') as f:
        f.write(str(threshold))
    print(f"Threshold ({threshold}) saved to {threshold_path}")


def load_model_and_threshold(model_path='xgb_best_model.joblib', threshold_path='threshold.txt'):
    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load threshold
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    print(f"Threshold ({threshold}) loaded from {threshold_path}")

    return model, threshold


def framingham_risk_score(age, total_chol, hdl_chol, sbp, is_treated_bp, smoker, diabetes, sex):
    """
    Compute Framingham 10-year risk of CVD.
    
    Parameters:
    - age: years
    - total_chol: mg/dL
    - hdl_chol: mg/dL
    - sbp: systolic BP (mm Hg)
    - is_treated_bp: 1 if on BP meds, else 0
    - smoker: 1 if current smoker, else 0
    - diabetes: 1 if diabetic, else 0
    - sex: 'M' or 'F'

    Returns:
    - 10-year CVD risk (probability between 0 and 1)
    """
    
    if sex == 'M':
        # Coefficients for Men
        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_sbp = np.log(sbp)
        
        coeffs = {
            'age': 3.06117,
            'total_chol': 1.12370,
            'hdl_chol': -0.93263,
            'sbp_treated': 1.99881,
            'sbp_untreated': 1.93303,
            'smoker': 0.65451,
            'diabetes': 0.57367,
            'mean': 23.9802,
            'baseline_survival': 0.88936
        }
        
    else:
        # Coefficients for Women
        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_sbp = np.log(sbp)
        
        coeffs = {
            'age': 2.32888,
            'total_chol': 1.20904,
            'hdl_chol': -0.70833,
            'sbp_treated': 2.82263,
            'sbp_untreated': 2.76157,
            'smoker': 0.52873,
            'diabetes': 0.69154,
            'mean': 26.1931,
            'baseline_survival': 0.95012
        }
    
    sbp_coeff = coeffs['sbp_treated'] if is_treated_bp else coeffs['sbp_untreated']
    
    # Linear combination of risk factors
    risk_score = (
        coeffs['age'] * ln_age
        + coeffs['total_chol'] * ln_total_chol
        + coeffs['hdl_chol'] * ln_hdl_chol
        + sbp_coeff * ln_sbp
        + coeffs['smoker'] * smoker
        + coeffs['diabetes'] * diabetes
    )
    
    # Compute 10-year risk
    risk = 1 - coeffs['baseline_survival'] ** np.exp(risk_score - coeffs['mean'])
    return risk
