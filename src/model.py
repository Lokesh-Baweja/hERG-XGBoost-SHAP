from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    return rf, xgb




