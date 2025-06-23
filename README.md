# Health Risk Prediction Using NHANES Data

## Project Overview

In this project, I built a machine learning model to estimate an individual's risk of serious illness (e.g., cardiovascular disease) using health survey data (NHANES). Such models can inform life insurance underwriting, public health interventions, and personal wellness programs.

## Problem Statement

Life insurers and healthcare organizations often need to assess health risk based on limited clinical and lifestyle data. My goal is to develop an interpretable ML model that provides calibrated risk estimates using easily collected features.

## Data

- Source: [NHANES public dataset](https://www.cdc.gov/nchs/nhanes/index.htm)
- Population: U.S. adults, 18+
- Features: Demographics, biometrics, lifestyle factors, clinical measures
- Target: High/low risk (binary)

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
## Model Performance

Final model: XGBoost Classifier

Selected threshold: **0.45**  
Rationale: Chosen to achieve **Recall ≥ 90%** (priority: catching at-risk patients)

Metrics at threshold 0.41:

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

health-risk-ml/
├── data/                             # Raw or processed data (if needed)
├── models/                           # Saved models and thresholds
│   ├── best_model.joblib
|   ├── xgb_best_model.joblib
│   ├── threshold.txt
├── notebooks/                        # EDA and ML notebooks
|   ├── 01_eda.ipynb
|   ├── 02_feature_engineering.ipynb
|   ├── 03_modeling.ipynb
├── src/                              # ML pipeline scripts
│   ├── data_prep.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── utils.py
├── environment.yml                   # Conda environment
└── README.md

## How to Run
Clone Repo and install env
```bash
conda env create -f environment.yml
conda activate health-risk-prediction
```

Train Model
```bash
python src/train_model.py
```

Evaluate Model
```bash
python src/evaluate_model.py --target_recall 0.9
```

Run Full Project
```bash
chmod +x run_all.sh
./run_all.sh
```
