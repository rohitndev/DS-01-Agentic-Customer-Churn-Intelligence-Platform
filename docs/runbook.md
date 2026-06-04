# Runbook — DS-01 Churn Intelligence Platform

## Local Development Quick Start

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Validate data
python -m src.ingestion.pipeline

# 3. Engineer features
python -m src.features.engineering

# 4. Train models (~2 min)
python -m src.models.trainer

# 5. Run tests
pytest tests/unit/ -v

# 6. Start API
uvicorn api.server:app --reload

# 7. View MLflow runs
mlflow ui
# Open http://127.0.0.1:5000
```

## Retrain Models Manually

```powershell
python -m src.models.trainer
```

## Run Full Pipeline (Airflow-style, local)

```powershell
python pipelines/daily_scoring_dag.py
python pipelines/drift_check_dag.py
```

## Common Issues

### "Models not found" (503 error)
Run `python -m src.models.trainer` first.

### "GROQ_API_KEY not set" (503 error on /agent)
Ensure `.env` file exists with `GROQ_API_KEY=gsk_...`

### `pkg_resources` ImportError
Run: `pip install "setuptools==69.5.1"`

### Groq model decommissioned error
Update `GROQ_MODEL` in `src/agent/retention_agent.py`.
Check current models at https://console.groq.com/docs/models

## GCP Deployment

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy via CI/CD
git push origin main          # triggers ci.yml + cd.yml automatically

# Manual deploy
bash deployment/ecr_push.sh   # (adapt for Artifact Registry)
gcloud run deploy ds01-churn-api \
  --image REGION-docker.pkg.dev/PROJECT/ds01-churn-api/ds01-churn-api:latest \
  --region asia-south1 --allow-unauthenticated
```

## Monitoring

| Signal | Threshold | Action |
|--------|-----------|--------|
| Drift share | > 20% | Trigger weekly retrain DAG |
| Model AUC | < 0.78 | Emergency retrain + rollback |
| API p99 latency | > 3s | Check Groq rate limits |
| Error rate | > 5% | Page on-call, check logs |
