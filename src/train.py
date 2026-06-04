"""
train.py — Train XGBoost + LSTM ensemble, log to MLflow, save models.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Make src/ importable when run directly
sys.path.insert(0, os.path.dirname(__file__))
from ingestion import load_and_validate
from features import build_feature_matrix

MODELS_DIR = "models"
SEQUENCE_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
SEQ_LEN = 3          # window length for the LSTM "sequence"
LSTM_EPOCHS = 20
LSTM_HIDDEN = 32
BATCH_SIZE = 64
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# LSTM definition
# ---------------------------------------------------------------------------

class ChurnLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)      # h_n: (1, batch, hidden)
        out = self.fc(h_n.squeeze(0))   # (batch, 1)
        return self.sigmoid(out).squeeze(1)


# ---------------------------------------------------------------------------
# Helper: build fake sequences for LSTM
# ---------------------------------------------------------------------------

def _build_lstm_sequences(X: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    The Telco dataset has no time dimension, so we simulate sequences by
    stacking [col/1, col/2, col/3] windows from the sequence columns.
    Each row produces one sequence of shape (seq_len, n_features).
    """
    cols = [c for c in SEQUENCE_COLS if c in X.columns]
    vals = X[cols].values.astype(np.float32)
    scaler = StandardScaler()
    vals = scaler.fit_transform(vals)
    # Repeat the single row seq_len times to form a (seq_len, n_features) window
    seqs = np.stack([vals] * seq_len, axis=1)  # (N, seq_len, n_features)
    return seqs, scaler


# ---------------------------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val):
    print("[train] Training XGBoost...")
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Train LSTM
# ---------------------------------------------------------------------------

def train_lstm(X_train, y_train, X_val, y_val):
    print("[train] Training LSTM...")
    seqs_train, scaler = _build_lstm_sequences(X_train)
    seqs_val, _ = _build_lstm_sequences(X_val)
    # Use the same scaler — fit only on train, apply to val
    cols = [c for c in SEQUENCE_COLS if c in X_train.columns]
    scaler.fit(X_train[cols].values.astype(np.float32))
    seqs_train = np.stack(
        [scaler.transform(X_train[cols].values.astype(np.float32))] * SEQ_LEN, axis=1
    )
    seqs_val = np.stack(
        [scaler.transform(X_val[cols].values.astype(np.float32))] * SEQ_LEN, axis=1
    )

    t_train = torch.tensor(seqs_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.values, dtype=torch.float32)
    t_val = torch.tensor(seqs_val, dtype=torch.float32)
    y_v = torch.tensor(y_val.values, dtype=torch.float32)

    loader = DataLoader(TensorDataset(t_train, y_t), batch_size=BATCH_SIZE, shuffle=True)

    model = ChurnLSTM(input_size=len(cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(LSTM_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(t_val).numpy()
    auc = roc_auc_score(y_val, val_preds)
    print(f"[train] LSTM val AUC: {auc:.4f}")
    return model, scaler


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_and_validate(data_path)
    X, y = build_feature_matrix(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("ds01-churn")
    with mlflow.start_run(run_name="ensemble_run"):

        # --- XGBoost ---
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]

        # --- LSTM ---
        lstm_model, lstm_scaler = train_lstm(X_train, y_train, X_val, y_val)
        cols = [c for c in SEQUENCE_COLS if c in X_val.columns]
        seqs_val = np.stack(
            [lstm_scaler.transform(X_val[cols].values.astype(np.float32))] * SEQ_LEN,
            axis=1,
        )
        with torch.no_grad():
            lstm_proba = lstm_model(torch.tensor(seqs_val, dtype=torch.float32)).numpy()

        # --- Ensemble ---
        ensemble_proba = 0.6 * xgb_proba + 0.4 * lstm_proba
        preds = (ensemble_proba >= 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, ensemble_proba)
        f1 = f1_score(y_val, preds)

        mlflow.log_param("xgb_weight", 0.6)
        mlflow.log_param("lstm_weight", 0.4)
        mlflow.log_param("lstm_epochs", LSTM_EPOCHS)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_f1", f1)

        print(f"\n[train] ===== Final Ensemble Metrics =====")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC      : {auc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"==========================================\n")

        # --- Save models ---
        xgb_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
        scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.joblib")
        feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.joblib")

        joblib.dump(xgb_model, xgb_path)
        torch.save(lstm_model.state_dict(), lstm_path)
        joblib.dump(lstm_scaler, scaler_path)
        joblib.dump(list(X_train.columns), feature_cols_path)

        print(f"[train] Models saved to {MODELS_DIR}/")
        return auc


if __name__ == "__main__":
    train()
