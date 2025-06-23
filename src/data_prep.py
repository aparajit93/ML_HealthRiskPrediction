import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import framingham_risk_score
import numpy as np

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
files = {
    "demographics": "DEMO_J.XPT",
    "cholesterol": "TCHOL_J.XPT",
    "hdl": "HDL_J.XPT",
    "blood_pressure": "BPX_J.XPT",
    "blood_cholesterol": "BPQ_J.XPT",
    "diabetes_q": "DIQ_J.XPT",
    "glucose": "GLU_J.XPT",
    "smoking": "SMQ_J.XPT",
    "body_measures": "BMX_J.XPT"
}

def load_and_prepare_data(test_size = 0.3, random_state = 42):
    dfs = {}
    for name, filename in files.items():
        dfs[name] = pd.read_sas(BASE_URL + filename)
    
    df = dfs['demographics']
    for name in dfs.keys():
        if name == 'demographics':
            continue
        df = df.merge(dfs[name], on='SEQN', how='inner')

    df = df.rename(columns=
               {
                   'RIDAGEYR' : 'age',
                   'RIAGENDR' : 'sex',
                   'LBXTC' : 'total_chol',
                   'LBDHDD' : 'hdl_chol',
                   'BPQ050A' : 'bp_treated',
                   'DIQ010' : 'diabetes',
                   'SMQ020' : 'ever_smoked',
                   'SMQ040' : 'current_smoke'
               })
    
    # Calculate mean blood pressure (across 3 readings)
    df['sbp'] = df[['BPXSY1', 'BPXSY2', 'BPXSY3']].mean(axis=1)

    # Convert sex to string
    df['sex'] = df['sex'].map({1: 'M', 2: 'F'})

    # Convert bp_treated, diabetes to binary
    df['bp_treated'] = df['bp_treated'].map({1: 1, 2: 0})
    df['diabetes'] = df['diabetes'].map({1: 1, 2: 0})

    # Define smoker: 1 if current smoker
    df['smoker'] = df['current_smoke'].map({1: 1, 2: 0, 7: 0, 9: 0})

    # Final columns to be used
    columns_needed = [
        'SEQN', 'age', 'sex', 'total_chol', 'hdl_chol',
        'sbp', 'bp_treated', 'smoker', 'diabetes'
    ]

    df = df[columns_needed]

    df = df.dropna(subset=[
        'age', 'sex', 'total_chol', 'hdl_chol',
        'sbp', 'bp_treated', 'diabetes'
    ])

    df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])

    df = df.reset_index(drop=True)

    df['framingham_risk'] = df.apply(
        lambda row: framingham_risk_score(
            row['age'],
            row['total_chol'],
            row['hdl_chol'],
            row['sbp'],
            row['bp_treated'],
            row['smoker'],
            row['diabetes'],
            row['sex']
        ),
    axis=1
    )
    
    # Define high risk > 20% 10-year risk
    df['high_risk'] = (df['framingham_risk'] > 0.20).astype(int)

    # Features for Farmingham risk factor. These will be the predictor labels
    features = [ 'age', 'total_chol', 'hdl_chol', 'sbp', 'bp_treated', 'smoker', 'diabetes']
    x = df[features]
    y = df['high_risk'] # Target variable. If someone is at high risk or not

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state= random_state, stratify=y)

    return x_train, x_test, y_train, y_test