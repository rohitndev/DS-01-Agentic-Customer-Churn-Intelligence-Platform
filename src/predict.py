"""
predict.py — Load saved models and run ensemble prediction with SHAP explanations.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features, CAT_COLS, SERVICE_COLS

MODELS_DIR = "models"
SEQUENCE_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
SEQ_LEN = 3


# ---------------------------------------------------------------------------
# LSTM must be defined here to load state_dict (same architecture as train.py)
# ---------------------------------------------------------------------------

import torch.nn as nn

class ChurnLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out).squeeze(1)


def _load_models():
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    lstm_model = ChurnLSTM(input_size=len(SEQUENCE_COLS))
    lstm_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "lstm_model.pt"), map_location="cpu")
    )
    lstm_model.eval()
    return xgb_model, lstm_model, scaler, feature_cols


def _preprocess_single(record: dict, feature_cols: list) -> pd.DataFrame:
    """Convert a raw customer dict into the aligned feature DataFrame."""
    df = pd.DataFrame([record])
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    df = engineer_features(df)

    # One-hot encode
    df_enc = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns])

    # Drop non-feature columns
    drop = ["customerID", "Churn"] + SERVICE_COLS
    df_enc = df_enc.drop(columns=[c for c in drop if c in df_enc.columns])
    df_enc = df_enc.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Align to training feature columns (add missing with 0, drop extras)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols]
    return df_enc


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def predict(record: dict) -> dict:
    """
    Accept a customer record dict, return:
      churn_probability, risk_label, top_shap_features (list of dicts)
    """
    xgb_model, lstm_model, scaler, feature_cols = _load_models()

    X = _preprocess_single(record, feature_cols)

    # --- XGBoost prediction ---
    xgb_proba = float(xgb_model.predict_proba(X)[:, 1][0])

    # --- LSTM prediction ---
    seq_cols = [c for c in SEQUENCE_COLS if c in X.columns]
    seq_vals = scaler.transform(X[seq_cols].values.astype(np.float32))
    seq_tensor = torch.tensor(
        np.stack([seq_vals] * SEQ_LEN, axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        lstm_proba = float(lstm_model(seq_tensor).numpy()[0])

    ensemble_prob = 0.6 * xgb_proba + 0.4 * lstm_proba

    # --- SHAP on XGBoost (tree-based, fast) ---
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    # shap_values shape: (1, n_features) for binary XGB
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    shap_series = pd.Series(sv, index=feature_cols).abs().sort_values(ascending=False)
    top3 = [
        {"feature": feat, "shap_value": round(float(sv[feature_cols.index(feat)]), 4)}
        for feat in shap_series.head(3).index
    ]

    result = {
        "churn_probability": round(ensemble_prob, 4),
        "risk_label": _risk_label(ensemble_prob),
        "top_shap_features": top3,
        "xgb_probability": round(xgb_proba, 4),
        "lstm_probability": round(lstm_proba, 4),
    }
    return result


if __name__ == "__main__":
    # Sample customer for quick test
    sample = {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.0,
    }
    result = predict(sample)
    print("\n===== Prediction Result =====")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=============================")
