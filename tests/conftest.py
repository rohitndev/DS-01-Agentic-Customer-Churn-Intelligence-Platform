"""
tests/conftest.py — Shared pytest fixtures for unit and integration tests.
"""

import os
import sys
import pytest
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAW_CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="session")
def sample_customer() -> dict:
    """A realistic customer record for prediction tests."""
    return {
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


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Load and return the raw dataset (session-scoped for speed)."""
    if not os.path.exists(RAW_CSV):
        pytest.skip(f"Dataset not found at {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(subset=["TotalCharges"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def feature_matrix(raw_df):
    from src.features.engineering import build_feature_matrix
    return build_feature_matrix(raw_df)
