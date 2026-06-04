"""
tests/unit/test_features.py — Unit tests for feature engineering.
"""

import pandas as pd
import pytest
from src.features.engineering import engineer_features, build_feature_matrix


@pytest.fixture
def minimal_df():
    return pd.DataFrame([{
        "customerID": "X-001", "gender": "Male", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 24,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.0, "TotalCharges": 1440.0, "Churn": "No",
    }])


def test_clv_calculation(minimal_df):
    out = engineer_features(minimal_df)
    assert out.loc[0, "CLV"] == pytest.approx(60.0 * 24)


def test_charge_per_tenure(minimal_df):
    out = engineer_features(minimal_df)
    expected = 60.0 / (24 + 1)
    assert out.loc[0, "charge_per_tenure"] == pytest.approx(expected)


def test_num_services(minimal_df):
    out = engineer_features(minimal_df)
    # Active services: PhoneService, OnlineSecurity, OnlineBackup, StreamingTV = 4
    assert out.loc[0, "num_services"] == 4


def test_is_high_value_false(minimal_df):
    out = engineer_features(minimal_df)
    # CLV = 1440 < 2000 → is_high_value = 0
    assert out.loc[0, "is_high_value"] == 0


def test_is_high_value_true(minimal_df):
    minimal_df = minimal_df.copy()
    minimal_df["tenure"] = 60
    minimal_df["MonthlyCharges"] = 80.0
    out = engineer_features(minimal_df)
    # CLV = 80 * 60 = 4800 > 2000
    assert out.loc[0, "is_high_value"] == 1


def test_build_feature_matrix_shape(raw_df):
    X, y = build_feature_matrix(raw_df)
    assert X.shape[0] == len(raw_df)
    assert len(y) == len(raw_df)
    assert set(y.unique()) <= {0, 1}


def test_feature_matrix_no_nulls(raw_df):
    X, _ = build_feature_matrix(raw_df)
    assert X.isnull().sum().sum() == 0, "Feature matrix must have no nulls"


def test_feature_matrix_all_numeric(raw_df):
    X, _ = build_feature_matrix(raw_df)
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is not numeric"
