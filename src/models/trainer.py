"""
src/models/trainer.py — Entry point: run full training pipeline end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingestion.pipeline import load_and_validate
from src.features.engineering import build_feature_matrix
from src.models.ensemble import train_ensemble


def main(data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = load_and_validate(data_path)
    X, y = build_feature_matrix(df)
    auc = train_ensemble(X, y)
    print(f"[trainer] Final AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    main()
