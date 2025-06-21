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

## Methods

- **EDA**: Exploratory analysis of missingness, distribution of key health variables
- **Feature Engineering**: BMI, cholesterol ratios, blood pressure categories, lifestyle aggregates
- **Models**:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost
- **Explainability**: SHAP values to understand model predictions
- **Fairness**: Checked model behavior across age, gender, ethnicity groups

## Results

- AUC-ROC: 0.85 (XGBoost model)
- Calibration: Model produces well-calibrated probabilities
- Key Predictors: Age, systolic BP, total cholesterol, HDL cholesterol, smoking status

## Next Steps

- Deploy as interactive app (Streamlit prototype)
- Test on newer NHANES cohorts
- Investigate potential biases in predictions

## Repository Structure

