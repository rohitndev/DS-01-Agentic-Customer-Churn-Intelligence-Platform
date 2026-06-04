"""
api/schemas.py — Pydantic request/response models for the FastAPI server
and Lambda handler.
"""

from typing import Any
from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    customerID:       str   = Field("UNKNOWN", description="Unique customer identifier")
    gender:           str   = "Male"
    SeniorCitizen:    int   = Field(0, ge=0, le=1)
    Partner:          str   = "No"
    Dependents:       str   = "No"
    tenure:           int   = Field(1, ge=0, description="Months with the company")
    PhoneService:     str   = "Yes"
    MultipleLines:    str   = "No"
    InternetService:  str   = "Fiber optic"
    OnlineSecurity:   str   = "No"
    OnlineBackup:     str   = "No"
    DeviceProtection: str   = "No"
    TechSupport:      str   = "No"
    StreamingTV:      str   = "No"
    StreamingMovies:  str   = "No"
    Contract:         str   = "Month-to-month"
    PaperlessBilling: str   = "Yes"
    PaymentMethod:    str   = "Electronic check"
    MonthlyCharges:   float = Field(70.0, gt=0)
    TotalCharges:     float = Field(70.0, ge=0)


class SHAPFeature(BaseModel):
    feature:    str
    shap_value: float
    direction:  str = ""


class PredictResponse(BaseModel):
    churn_probability:  float
    risk_label:         str
    top_shap_features:  list[dict[str, Any]]
    xgb_probability:    float
    lstm_probability:   float


class AgentResponse(BaseModel):
    retention_message:  str
    recommended_action: str
    churn_score:        float
    risk_label:         str
    nl_reason:          str = ""


class DriftResponse(BaseModel):
    total_features:   int
    drifted_count:    int
    drift_share:      float
    drifted_features: list[dict[str, Any]]
    report_saved_to:  str
