"""
features.py — Engineer features from the IBM Telco CSV for model training.
"""

import pandas as pd
from ingestion import load_and_validate


# Binary service columns to count active services
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Categorical columns to one-hot encode
CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _count_services(df: pd.DataFrame) -> pd.Series:
    """Count how many optional services a customer has active."""
    svc = df[SERVICE_COLS].apply(
        lambda col: (col == "Yes").astype(int)
    )
    return svc.sum(axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features and return a copy of the DataFrame.

    New columns:
      CLV                — Customer Lifetime Value proxy
      charge_per_tenure  — Monthly cost efficiency
      num_services       — Total active add-on services
      is_high_value      — Binary flag for CLV > 2000
    """
    df = df.copy()

    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["num_services"] = _count_services(df)
    df["is_high_value"] = (df["CLV"] > 2000).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) where X is the model-ready feature matrix
    and y is the binary Churn label (1 = Yes, 0 = No).
    """
    df = engineer_features(df)

    # Target
    y = (df["Churn"] == "Yes").astype(int)

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

    # SeniorCitizen is already 0/1 numeric
    # Drop columns not useful for modelling
    drop_cols = ["customerID", "Churn"] + SERVICE_COLS
    drop_cols = [c for c in drop_cols if c in df_encoded.columns]
    X = df_encoded.drop(columns=drop_cols)

    # Ensure all remaining columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[features] Feature matrix shape: {X.shape}")
    print(f"[features] Churn rate: {y.mean():.2%}")
    return X, y


if __name__ == "__main__":
    df = load_and_validate()
    X, y = build_feature_matrix(df)
    print(X.head())
    print("Feature columns:", list(X.columns))
