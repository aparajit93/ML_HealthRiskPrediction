# Health Risk Prediction Using NHANES Data

## Project Overview

In this project, I built a machine learning model to estimate an individual's risk of serious illness (e.g., cardiovascular disease) using health survey data (NHANES). Such models can inform life insurance underwriting, public health interventions, and personal wellness programs.

## Problem Statement

Life insurers and healthcare organizations often need to assess health risk based on limited clinical and lifestyle data. My goal is to develop an interpretable ML model that provides calibrated risk estimates using easily collected features.

## Data

- Source: [NHANES public dataset](https://www.cdc.gov/nchs/nhanes/index.htm)
- Population: U.S. adults, 18+
- Features: Demographics, biometrics, lifestyle factors, clinical measures
- Target: Calculated Framingham 10-year CVD risk score (continuous) or high/low risk (binary)

## Handling Missing Data

Several variables in the NHANES dataset had missing values:

| Variable          | % Missing | Imputation Method  |
|-------------------|-----------|-------------------|
| Total cholesterol | 6%        | Rows dropped if missing |
| HDL cholesterol   | 6%        | Rows dropped if missing |
| Systolic BP       | 5%        | Rows dropped if missing |
| If on BP meds     | 67%       | Rows dropped if missing |
| Smoker            | 82%       | Mode imputation |
| Diabetes          | 3%        | Mode imputation |
| Sex, Age, BP treatment | < 1% | Rows dropped if missing |

After imputation, 40% of the original dataset was retained.


## Methods

- **EDA**: Exploratory analysis of missingness, distribution of key health variables
- **Feature Engineering**: BMI, cholesterol ratios, blood pressure categories, lifestyle aggregates
- **Models**:
    - XGBoost
- **Explainability**: SHAP values to understand model predictions
- **Fairness**: Checked model behavior across age, gender, ethnicity groups

## Model Performance

Final model: XGBoost Classifier

Selected threshold: **0.45**  
Rationale: Chosen to achieve **Recall ≥ 90%** (priority: catching at-risk patients)

Metrics at threshold 0.32:

- Recall: 0.90
- Precision: 0.88
- F1 Score: 0.89
- AUC: 0.93

Confusion Matrix:
- True Positives: 127
- False Negatives: 14
- False Positives: 18
- True Negatives: 70

## Results

- AUC-ROC: 0.93 (XGBoost model)
- Calibration: Model produces well-calibrated probabilities
- Key Predictors: Age, systolic BP, total cholesterol, HDL cholesterol, smoking status

## Next Steps

- Deploy as interactive app (Streamlit prototype)
- Test on newer NHANES cohorts
- Investigate potential biases in predictions

## Repository Structure

