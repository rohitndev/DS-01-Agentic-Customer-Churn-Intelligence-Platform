"""
api/server.py — FastAPI inference server.

Run:  uvicorn api.server:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from api.schemas import CustomerRequest, PredictResponse, AgentResponse, DriftResponse
from src.models.predictor import predict
from src.agent.retention_agent import run_retention_agent
from mlops.drift_monitor import run_drift_report

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

app = FastAPI(
    title="DS-01 Churn Intelligence API",
    description=(
        "Agentic Customer Churn Intelligence Platform — "
        "XGBoost+LSTM ensemble, SHAP explanations, LangGraph retention agent."
    ),
    version="2.0.0",
)


@app.get("/")
def root():
    return {"service": "DS-01 Churn Intelligence API", "docs": "/docs", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_churn(customer: CustomerRequest):
    """Ensemble churn prediction + SHAP top-3 explanation."""
    try:
        return predict(customer.model_dump())
    except FileNotFoundError:
        raise HTTPException(503, "Models not found. Run `python -m src.models.trainer` first.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/agent", response_model=AgentResponse, tags=["Agent"])
def retention_agent(customer: CustomerRequest):
    """LangGraph agent: churn score → personalized retention message via Groq."""
    try:
        return run_retention_agent(customer.model_dump())
    except FileNotFoundError:
        raise HTTPException(503, "Models not found. Run the trainer first.")
    except ValueError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
def drift_report():
    """Evidently AI drift report: reference (70%) vs current (30%) split."""
    try:
        return run_drift_report(DATA_PATH)
    except FileNotFoundError:
        raise HTTPException(503, f"Dataset not found at {DATA_PATH}.")
    except Exception as e:
        raise HTTPException(500, str(e))
