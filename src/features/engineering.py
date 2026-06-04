"""
src/features/engineering.py — Feature engineering: CLV, charge efficiency,
service count, and one-hot encoding. Writes Gold Parquet layer.
"""

import os
import pandas as pd

GOLD = "data/processed/gold/telco_features.parquet"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CAT_COLS = [
    "gender", "Partner", "Dependents", "InternetService",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def _count_services(df: pd.DataFrame) -> pd.Series:
    svc = df[SERVICE_COLS].apply(lambda col: (col == "Yes").astype(int))
    return svc.sum(axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CLV, charge_per_tenure, num_services, is_high_value columns."""
    df = df.copy()
    df["CLV"]               = df["MonthlyCharges"] * df["tenure"]
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["num_services"]      = _count_services(df)
    df["is_high_value"]     = (df["CLV"] > 2000).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned and ready for model training."""
    df = engineer_features(df)
    y = (df["Churn"] == "Yes").astype(int)
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    drop = ["customerID", "Churn"] + SERVICE_COLS
    X = df_enc.drop(columns=[c for c in drop if c in df_enc.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"[features] Feature matrix: {X.shape} | Churn rate: {y.mean():.2%}")
    return X, y


def write_gold(X: pd.DataFrame, y: pd.Series) -> None:
    os.makedirs(os.path.dirname(GOLD), exist_ok=True)
    gold = X.copy()
    gold["Churn"] = y
    gold.to_parquet(GOLD, index=False)
    print(f"[features] Gold layer written -> {GOLD}")


if __name__ == "__main__":
    from src.ingestion.pipeline import load_and_validate
    df = load_and_validate()
    X, y = build_feature_matrix(df)
    write_gold(X, y)
    print(X.head())
