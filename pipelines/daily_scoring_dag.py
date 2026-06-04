"""
pipelines/daily_scoring_dag.py — Airflow DAG: daily batch churn scoring.

Schedule: runs every day at 02:00 UTC.
Tasks:
  1. ingest    → validate fresh CSV export
  2. score     → run ensemble predictor on all customers
  3. alert     → send Slack alerts for High-risk customers
  4. crm_log   → write actions to CRM (SQLite locally / CRM API in prod)

To run locally without Airflow:
  python pipelines/daily_scoring_dag.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Airflow imports (guarded so file is importable without Airflow) ---
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

import pandas as pd
from src.ingestion.pipeline import load_and_validate
from src.features.engineering import build_feature_matrix
from src.models.predictor import predict
from src.agent.slack_notifier import send_alert
from src.agent.crm_client import log_retention_action

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
HIGH_RISK_THRESHOLD = 0.70


def task_ingest(**_):
    df = load_and_validate(DATA_PATH)
    print(f"[daily_dag] Ingested {len(df)} rows")
    return len(df)


def task_score(**_):
    from src.ingestion.pipeline import load_csv
    df = load_csv(DATA_PATH)
    results = []
    for _, row in df.iterrows():
        try:
            r = predict(row.to_dict())
            r["customerID"] = row["customerID"]
            results.append(r)
        except Exception as e:
            print(f"[daily_dag] Skipping {row['customerID']}: {e}")
    high_risk = [r for r in results if r["churn_probability"] >= HIGH_RISK_THRESHOLD]
    print(f"[daily_dag] Scored {len(results)} customers | High-risk: {len(high_risk)}")
    return high_risk


def task_alert(**context):
    high_risk = context["task_instance"].xcom_pull(task_ids="score") if HAS_AIRFLOW else task_score()
    for r in high_risk:
        send_alert(
            customer_id=r["customerID"],
            risk_label=r["risk_label"],
            churn_score=r["churn_probability"],
            action="escalate" if r["churn_probability"] >= 0.85 else "callback",
        )


def task_crm_log(**context):
    high_risk = context["task_instance"].xcom_pull(task_ids="score") if HAS_AIRFLOW else task_score()
    for r in high_risk:
        log_retention_action(
            customer_id=r["customerID"],
            action="escalate" if r["churn_probability"] >= 0.85 else "discount",
            churn_score=r["churn_probability"],
            risk_label=r["risk_label"],
            message="Auto-logged by daily scoring DAG",
        )


# --- DAG definition (only registered when Airflow is installed) ---
if HAS_AIRFLOW:
    default_args = {
        "owner":           "ds01-team",
        "retries":         2,
        "retry_delay":     timedelta(minutes=5),
        "email_on_failure": False,
    }
    with DAG(
        dag_id="ds01_daily_scoring",
        start_date=datetime(2025, 1, 1),
        schedule_interval="0 2 * * *",
        catchup=False,
        default_args=default_args,
        tags=["churn", "scoring"],
    ) as dag:
        t_ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
        t_score  = PythonOperator(task_id="score",  python_callable=task_score)
        t_alert  = PythonOperator(task_id="alert",  python_callable=task_alert)
        t_crm    = PythonOperator(task_id="crm_log",python_callable=task_crm_log)
        t_ingest >> t_score >> [t_alert, t_crm]


if __name__ == "__main__":
    print("[daily_dag] Running locally (no Airflow)")
    task_ingest()
    high = task_score()
    print(f"[daily_dag] High-risk customers: {len(high)}")
    for r in high[:3]:
        print(f"  {r['customerID']} — {r['churn_probability']:.2%} {r['risk_label']}")
