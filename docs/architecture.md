# Architecture — DS-01 Agentic Customer Churn Intelligence Platform

## High-Level Data Flow

```
IBM Telco CSV
     │
     ▼
[src/ingestion/pipeline.py]
  Great Expectations → 8 validation checks
  Bronze Parquet (raw)
  Silver Parquet (cleaned)
     │
     ▼
[src/features/engineering.py]
  CLV · charge_per_tenure · num_services · is_high_value
  One-hot encoding
  Gold Parquet (model-ready)
     │
     ├──────────────────────────────┐
     ▼                              ▼
[src/models/xgb_trainer.py]  [src/models/lstm_trainer.py]
  XGBoost (300 trees)         PyTorch LSTM (seq_len=3)
  AUC ~0.83                   AUC ~0.81
     │                              │
     └──────────┬───────────────────┘
                ▼
     [src/models/ensemble.py]
       score = 0.6·XGB + 0.4·LSTM
       MLflow logging
                │
                ├──────────────────────────────────┐
                ▼                                  ▼
   [src/explainability/shap_explainer.py]   [src/explainability/nl_reason_generator.py]
     SHAP TreeExplainer                       Plain-English churn explanation
                │                                  │
                └────────────┬─────────────────────┘
                             ▼
              [src/agent/retention_agent.py]  ← LangGraph
                Node 1: Analyze (predict + SHAP)
                Node 2: Draft  (Groq LLaMA-3.1-8b-instant)
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          [src/agent/crm_client.py]  [src/agent/slack_notifier.py]
            SQLite / CRM API          Webhook alert
                             │
                             ▼
                    [api/server.py] (FastAPI)
                    POST /predict
                    POST /agent
                    GET  /drift
                             │
                             ▼
                    [mlops/drift_monitor.py]
                    Evidently DataDriftPreset
                    Reference 70% / Current 30%
```

## Medallion Data Architecture

| Layer  | Path                              | Format  | Content                          |
|--------|-----------------------------------|---------|----------------------------------|
| Bronze | `data/processed/bronze/`          | Parquet | Raw CSV as-is                    |
| Silver | `data/processed/silver/`          | Parquet | Cleaned (nulls removed)          |
| Gold   | `data/processed/gold/`            | Parquet | Feature-engineered, model-ready  |

## Model Architecture

### XGBoost
- 300 estimators, max_depth=5, lr=0.05
- Subsample 0.8, colsample 0.8
- Binary classification (Churn Yes/No)

### LSTM
- Input: 3-step sequence of [tenure, MonthlyCharges, TotalCharges]
- Hidden: 32 units, single layer
- Output: sigmoid probability
- Trained 20 epochs, Adam lr=1e-3

### Ensemble
```
churn_score = 0.6 × XGBoost_proba + 0.4 × LSTM_proba
```

## Cloud Deployment (GCP)

```
Developer
    │ git push main
    ▼
GitHub Actions (ci.yml)
  → Lint (Ruff)
  → Unit tests
  → Integration tests
    │ pass
    ▼
GitHub Actions (cd.yml)
  → Train models
  → AUC gate (≥ 0.80)
  → Upload to GCS
  → Docker build → Artifact Registry
  → gcloud run deploy
    │
    ▼
Cloud Run (asia-south1)
  FastAPI → uvicorn
  Auto-scaling 0–10 instances
```
