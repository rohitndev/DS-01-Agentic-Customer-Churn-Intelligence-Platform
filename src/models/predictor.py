"""
src/models/predictor.py — Load ensemble models and score a single customer.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import torch

from .lstm_trainer import ChurnLSTM, SEQ_COLS, SEQ_LEN
from ..features.engineering import engineer_features, CAT_COLS, SERVICE_COLS

MODELS_DIR        = "models"
XGB_PATH          = "models/xgb_model.joblib"
LSTM_PATH         = "models/lstm_model.pt"
SCALER_PATH       = "models/lstm_scaler.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"

XGB_WEIGHT  = 0.6
LSTM_WEIGHT = 0.4


def _load_all():
    xgb     = joblib.load(XGB_PATH)
    scaler  = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    lstm    = ChurnLSTM(input_size=len(SEQ_COLS))
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
    lstm.eval()
    return xgb, lstm, scaler, feat_cols


def _preprocess(record: dict, feature_cols: list) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    df = engineer_features(df)
    df_enc = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns])
    df_enc = df_enc.drop(columns=[c for c in ["customerID", "Churn"] + SERVICE_COLS if c in df_enc.columns])
    df_enc = df_enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    return df_enc[feature_cols]


def _risk_label(prob: float) -> str:
    return "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"


def predict(record: dict) -> dict:
    xgb, lstm, scaler, feat_cols = _load_all()
    X = _preprocess(record, feat_cols)

    xgb_proba  = float(xgb.predict_proba(X)[:, 1][0])
    seqs       = np.stack([scaler.transform(X[[c for c in SEQ_COLS if c in X.columns]].values.astype(np.float32))] * SEQ_LEN, axis=1)
    with torch.no_grad():
        lstm_proba = float(lstm(torch.tensor(seqs, dtype=torch.float32)).numpy()[0])

    prob = XGB_WEIGHT * xgb_proba + LSTM_WEIGHT * lstm_proba

    explainer   = shap.TreeExplainer(xgb)
    shap_vals   = explainer.shap_values(X)
    sv          = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_series = pd.Series(sv, index=feat_cols).abs().sort_values(ascending=False)
    top3 = [
        {
            "feature": f,
            "shap_value": round(float(sv[feat_cols.index(f)]), 4),
            "direction": "increases_churn" if sv[feat_cols.index(f)] > 0 else "decreases_churn",
        }
        for f in shap_series.head(3).index
    ]

    return {
        "churn_probability": round(prob, 4),
        "risk_label":        _risk_label(prob),
        "top_shap_features": top3,
        "xgb_probability":   round(xgb_proba, 4),
        "lstm_probability":  round(lstm_proba, 4),
    }
