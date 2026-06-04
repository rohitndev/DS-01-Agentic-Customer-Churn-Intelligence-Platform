"""
api.py — FastAPI REST endpoints for churn prediction, retention agent, and drift.

Run with:  uvicorn src.api:app --reload
"""

import os
import sys
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src/ modules are importable when uvicorn loads from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from predict import predict
from agent import run_retention_agent
from drift import run_drift_report

app = FastAPI(
    title="DS-01 Churn Intelligence API",
    description="Agentic Customer Churn Intelligence Platform — DS-01 College Project",
    version="1.0.0",
)

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CustomerRequest(BaseModel):
    customerID: str = "UNKNOWN"
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 70.0


class PredictResponse(BaseModel):
    churn_probability: float
    risk_label: str
    top_shap_features: list[dict[str, Any]]
    xgb_probability: float
    lstm_probability: float


class AgentResponse(BaseModel):
    retention_message: str
    recommended_action: str
    churn_score: float
    risk_label: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_count: int
    drift_share: float
    drifted_features: list[dict[str, Any]]
    report_saved_to: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "DS-01 Churn Intelligence API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictResponse)
def predict_churn(customer: CustomerRequest):
    """
    Run ensemble churn prediction (XGBoost + LSTM) and return SHAP explanations.
    """
    try:
        result = predict(customer.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Models not found. Run `python src/train.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent", response_model=AgentResponse)
def retention_agent(customer: CustomerRequest):
    """
    Run the LangGraph retention agent: predicts churn, then drafts
    a personalized retention message via Groq LLM.
    """
    try:
        result = run_retention_agent(customer.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Models not found. Run `python src/train.py` first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift", response_model=DriftResponse)
def drift_report():
    """
    Run Evidently AI drift report comparing reference vs current data.
    Returns which features have drifted and the overall drift share.
    """
    try:
        result = run_drift_report(DATA_PATH)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not found at {DATA_PATH}. Add the CSV to data/ folder.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
