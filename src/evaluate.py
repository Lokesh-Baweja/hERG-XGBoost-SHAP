import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_models(rf_model, xgb_model, X_test, y_test):
    rf_preds = rf_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    ensemble_preds = (rf_preds + xgb_preds) / 2

    rf_auc = roc_auc_score(y_test, rf_preds)
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_preds)

    print(f"RF AUC: {rf_auc:.4f}")
    print(f"XGB AUC: {xgb_auc:.4f}")

    return {
        "rf_auc": rf_auc,
        "xgb_auc": xgb_auc,
    }

