import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Step 1: Dynamically find the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(project_root, "models")

# Step 2: Create the models directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

def train_models(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    param_grid_rf = {
        'n_estimators': [120],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 9],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Tuning Random Forest...")
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    rf_best = grid_rf.best_estimator_
    print("Best RF Parameters:", grid_rf.best_params_)

    print("Tuning XGBoost...")
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    xgb_best = grid_xgb.best_estimator_
    print("Best XGB Parameters:", grid_xgb.best_params_)

    # Save models using absolute paths
    joblib.dump(rf_best, os.path.join(model_dir, "rf_best_model.pkl"))
    joblib.dump(xgb_best, os.path.join(model_dir, "xgb_best_model.pkl"))

    # Save hyperparameters
    with open(os.path.join(model_dir, "best_hyperparameters.txt"), "w") as f:
        f.write(f"Best RF Parameters: {grid_rf.best_params_}\n")
        f.write(f"Best XGB Parameters: {grid_xgb.best_params_}\n")

    return rf_best, xgb_best

