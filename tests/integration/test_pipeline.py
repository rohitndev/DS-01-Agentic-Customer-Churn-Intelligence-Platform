"""
tests/integration/test_pipeline.py — End-to-end pipeline integration tests.
Requires the dataset and trained models to be present.
"""

import os
import pytest


@pytest.mark.skipif(
    not os.path.exists("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    reason="Dataset not found",
)
def test_ingestion_returns_dataframe():
    from src.ingestion.pipeline import load_and_validate
    df = load_and_validate()
    assert len(df) > 7000
    assert "Churn" in df.columns
    assert "tenure" in df.columns


def test_feature_matrix_has_engineered_cols(raw_df):
    from src.features.engineering import engineer_features
    out = engineer_features(raw_df)
    assert "CLV" in out.columns
    assert "charge_per_tenure" in out.columns
    assert "num_services" in out.columns
    assert "is_high_value" in out.columns


@pytest.mark.skipif(
    not os.path.exists("models/xgb_model.joblib"),
    reason="Models not trained yet",
)
def test_prediction_pipeline(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    assert result["churn_probability"] > 0
    assert result["risk_label"] in {"High", "Medium", "Low"}


@pytest.mark.skipif(
    not os.path.exists("models/xgb_model.joblib"),
    reason="Models not trained yet",
)
def test_model_auc_regression(raw_df):
    """Ensure the trained model still meets the minimum AUC bar."""
    import joblib
    from sklearn.metrics import roc_auc_score
    from src.features.engineering import build_feature_matrix

    MIN_AUC = 0.78

    xgb = joblib.load("models/xgb_model.joblib")
    X, y = build_feature_matrix(raw_df)
    proba = xgb.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    assert auc >= MIN_AUC, f"Model AUC {auc:.4f} fell below regression threshold {MIN_AUC}"
