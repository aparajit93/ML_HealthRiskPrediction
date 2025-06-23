import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_and_prepare_data
from src.utils import save_model_and_threshold

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test = load_and_prepare_data()

model = xgb.XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    use_label_encoder = False,
    eval_metric = 'logloss'
)

# Grid search tuning
# Define parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

save_model_and_threshold(
    best_model,
    threshold=0.5,  # default threshold for now
    model_path='model\\xgb_best_model.joblib',
    threshold_path='model\\threshold.txt'
)

print("Training complete. Model and threshold saved.")