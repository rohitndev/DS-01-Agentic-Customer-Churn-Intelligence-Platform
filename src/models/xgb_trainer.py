"""
src/models/xgb_trainer.py — XGBoost classifier training with MLflow logging.
"""

import joblib
import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score

MODEL_PATH = "models/xgb_model.joblib"
RANDOM_STATE = 42


def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"[xgb_trainer] Val AUC: {auc:.4f}")
    mlflow.log_metric("xgb_val_auc", auc)
    return model, proba


def save_xgb(model) -> None:
    joblib.dump(model, MODEL_PATH)
    print(f"[xgb_trainer] Saved -> {MODEL_PATH}")


def load_xgb():
    return joblib.load(MODEL_PATH)
