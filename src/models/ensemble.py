"""
src/models/ensemble.py — Combine XGBoost + LSTM predictions into a
weighted ensemble score and run the full training pipeline.
"""

import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from .xgb_trainer import train_xgboost, save_xgb
from .lstm_trainer import train_lstm, save_lstm

FEATURE_COLS_PATH = "models/feature_cols.joblib"
XGB_WEIGHT = 0.6
LSTM_WEIGHT = 0.4
RANDOM_STATE = 42


def train_ensemble(X, y) -> float:
    os.makedirs("models", exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("ds01-churn")
    with mlflow.start_run(run_name="ensemble_run"):
        mlflow.log_param("xgb_weight", XGB_WEIGHT)
        mlflow.log_param("lstm_weight", LSTM_WEIGHT)

        print("[ensemble] Training XGBoost...")
        xgb_model, xgb_proba = train_xgboost(X_train, y_train, X_val, y_val)

        print("[ensemble] Training LSTM...")
        lstm_model, scaler, lstm_proba = train_lstm(X_train, y_train, X_val, y_val)

        ensemble_proba = XGB_WEIGHT * xgb_proba + LSTM_WEIGHT * lstm_proba
        preds = (ensemble_proba >= 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, ensemble_proba)
        f1  = f1_score(y_val, preds)

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_f1", f1)

        print(f"\n[ensemble] ===== Final Metrics =====")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC      : {auc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"=====================================\n")

        save_xgb(xgb_model)
        save_lstm(lstm_model, scaler)
        joblib.dump(list(X_train.columns), FEATURE_COLS_PATH)

        return auc
