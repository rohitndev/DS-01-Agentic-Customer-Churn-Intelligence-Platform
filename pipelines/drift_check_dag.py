"""
pipelines/drift_check_dag.py — Airflow DAG: daily drift detection.

Schedule: every day at 03:00 UTC (after daily scoring).
Tasks:
  1. run_drift → Evidently DataDriftPreset report
  2. alert     → Slack alert if drift share exceeds threshold
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

from mlops.drift_monitor import run_drift_report
from src.agent.slack_notifier import send_alert

DRIFT_ALERT_THRESHOLD = 0.20   # alert if >20% of features drift


def task_run_drift(**_) -> dict:
    return run_drift_report()


def task_alert_drift(**context):
    summary = (
        context["task_instance"].xcom_pull(task_ids="run_drift")
        if HAS_AIRFLOW else task_run_drift()
    )
    if summary["drift_share"] >= DRIFT_ALERT_THRESHOLD:
        send_alert(
            customer_id="SYSTEM",
            risk_label="High",
            churn_score=summary["drift_share"],
            action=f"Data drift alert: {summary['drifted_count']}/{summary['total_features']} features drifted",
        )


if HAS_AIRFLOW:
    default_args = {
        "owner":            "ds01-team",
        "retries":          1,
        "retry_delay":      timedelta(minutes=5),
        "email_on_failure": False,
    }
    with DAG(
        dag_id="ds01_drift_check",
        start_date=datetime(2025, 1, 1),
        schedule_interval="0 3 * * *",
        catchup=False,
        default_args=default_args,
        tags=["churn", "monitoring"],
    ) as dag:
        t_drift = PythonOperator(task_id="run_drift",   python_callable=task_run_drift)
        t_alert = PythonOperator(task_id="alert_drift", python_callable=task_alert_drift)
        t_drift >> t_alert


if __name__ == "__main__":
    summary = task_run_drift()
    print(f"[drift_dag] Drift share: {summary['drift_share']:.0%}")
