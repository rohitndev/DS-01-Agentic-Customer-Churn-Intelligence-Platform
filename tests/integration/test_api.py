"""
tests/integration/test_api.py — Integration tests for FastAPI endpoints.
Requires the API server to be running OR uses the TestClient directly.
"""

import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

CUSTOMER_PAYLOAD = {
    "customerID": "INTEG-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 420.0,
}


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DS-01" in resp.json()["service"]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_status_200():
    resp = client.post("/predict", json=CUSTOMER_PAYLOAD)
    assert resp.status_code == 200


def test_predict_response_schema():
    resp = client.post("/predict", json=CUSTOMER_PAYLOAD)
    body = resp.json()
    assert "churn_probability" in body
    assert "risk_label" in body
    assert "top_shap_features" in body
    assert isinstance(body["top_shap_features"], list)


def test_predict_probability_range():
    resp = client.post("/predict", json=CUSTOMER_PAYLOAD)
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_invalid_payload():
    resp = client.post("/predict", json={"tenure": -5, "MonthlyCharges": -10})
    # Pydantic validation should reject negative MonthlyCharges
    assert resp.status_code == 422
