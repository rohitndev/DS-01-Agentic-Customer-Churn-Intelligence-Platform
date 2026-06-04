"""
tests/unit/test_predictor.py — Unit tests for ensemble prediction output shape/types.
"""

import pytest


def test_predict_returns_required_keys(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    required = {"churn_probability", "risk_label", "top_shap_features",
                "xgb_probability", "lstm_probability"}
    assert required.issubset(result.keys())


def test_churn_probability_range(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_risk_label_valid(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    assert result["risk_label"] in {"High", "Medium", "Low"}


def test_shap_features_count(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    assert len(result["top_shap_features"]) == 3


def test_shap_feature_structure(sample_customer):
    from src.models.predictor import predict
    result = predict(sample_customer)
    for feat in result["top_shap_features"]:
        assert "feature" in feat
        assert "shap_value" in feat
        assert isinstance(feat["shap_value"], float)


def test_risk_label_consistency(sample_customer):
    from src.models.predictor import predict, _risk_label
    result = predict(sample_customer)
    assert result["risk_label"] == _risk_label(result["churn_probability"])
