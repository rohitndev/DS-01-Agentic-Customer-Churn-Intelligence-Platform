"""
tests/unit/test_explainability.py — Unit tests for SHAP explainer and NL reason generator.
"""

import pytest
from src.explainability.nl_reason_generator import generate_reason, _describe_feature


def test_generate_reason_with_drivers():
    shap_features = [
        {"feature": "Contract_Month-to-month", "shap_value": 0.58, "direction": "increases_churn"},
        {"feature": "tenure",                  "shap_value": -0.30, "direction": "decreases_churn"},
        {"feature": "MonthlyCharges",           "shap_value": 0.20, "direction": "increases_churn"},
    ]
    reason = generate_reason(shap_features)
    assert isinstance(reason, str)
    assert len(reason) > 20
    assert "churn" in reason.lower()


def test_generate_reason_empty():
    reason = generate_reason([])
    assert "No significant" in reason


def test_describe_feature_known():
    desc = _describe_feature("Contract_Month-to-month")
    assert "month-to-month" in desc.lower()


def test_describe_feature_unknown():
    desc = _describe_feature("some_unknown_feature_xyz")
    assert "some unknown feature xyz" in desc.lower()


def test_generate_reason_single_driver():
    shap_features = [
        {"feature": "InternetService_Fiber optic", "shap_value": 0.4, "direction": "increases_churn"}
    ]
    reason = generate_reason(shap_features)
    assert isinstance(reason, str)
    assert "churn" in reason.lower()
