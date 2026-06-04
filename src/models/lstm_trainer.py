"""
src/models/lstm_trainer.py — PyTorch LSTM classifier training.
"""

import numpy as np
import joblib
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LSTM_PATH   = "models/lstm_model.pt"
SCALER_PATH = "models/lstm_scaler.joblib"
SEQ_COLS    = ["tenure", "MonthlyCharges", "TotalCharges"]
SEQ_LEN     = 3
HIDDEN      = 32
EPOCHS      = 20
BATCH       = 64


class ChurnLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n.squeeze(0))).squeeze(1)


def _build_sequences(X, scaler: StandardScaler) -> np.ndarray:
    cols = [c for c in SEQ_COLS if c in X.columns]
    vals = scaler.transform(X[cols].values.astype(np.float32))
    return np.stack([vals] * SEQ_LEN, axis=1)


def train_lstm(X_train, y_train, X_val, y_val):
    cols = [c for c in SEQ_COLS if c in X_train.columns]
    scaler = StandardScaler()
    scaler.fit(X_train[cols].values.astype(np.float32))

    seqs_train = _build_sequences(X_train, scaler)
    seqs_val   = _build_sequences(X_val, scaler)

    t_train = torch.tensor(seqs_train, dtype=torch.float32)
    y_t     = torch.tensor(y_train.values, dtype=torch.float32)
    t_val   = torch.tensor(seqs_val, dtype=torch.float32)
    y_v     = torch.tensor(y_val.values, dtype=torch.float32)

    loader = DataLoader(TensorDataset(t_train, y_t), batch_size=BATCH, shuffle=True)
    model  = ChurnLSTM(input_size=len(cols))
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        proba = model(t_val).numpy()

    auc = roc_auc_score(y_v.numpy(), proba)
    print(f"[lstm_trainer] Val AUC: {auc:.4f}")
    mlflow.log_metric("lstm_val_auc", auc)
    return model, scaler, proba


def save_lstm(model: ChurnLSTM, scaler: StandardScaler) -> None:
    torch.save(model.state_dict(), LSTM_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[lstm_trainer] Saved -> {LSTM_PATH}, {SCALER_PATH}")


def load_lstm() -> tuple[ChurnLSTM, StandardScaler]:
    cols = SEQ_COLS
    model = ChurnLSTM(input_size=len(cols))
    model.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
