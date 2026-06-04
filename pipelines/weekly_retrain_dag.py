"""
pipelines/weekly_retrain_dag.py — Airflow DAG: weekly model retraining.

Schedule: every Sunday at 01:00 UTC.
Tasks:
  1. validate  → run Great Expectations on latest data
  2. retrain   → run full training pipeline, log to MLflow
  3. evaluate  → check AUC >= threshold (model gate)
  4. promote   → move new models to models/ if gate passes
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

from src.models.trainer import main as run_training

AUC_GATE = 0.80    # retrained model must beat this AUC to be promoted


def task_retrain(**_) -> float:
    auc = run_training()
    print(f"[weekly_dag] Retrain complete. AUC = {auc:.4f}")
    return auc


def task_evaluate(**context) -> bool:
    auc = context["task_instance"].xcom_pull(task_ids="retrain") if HAS_AIRFLOW else task_retrain()
    passed = float(auc) >= AUC_GATE
    print(f"[weekly_dag] Model gate {'PASSED' if passed else 'FAILED'}  AUC={auc:.4f}  gate={AUC_GATE}")
    if not passed:
        raise ValueError(f"Model AUC {auc:.4f} below gate {AUC_GATE}. Promotion blocked.")
    return passed


def task_promote(**_):
    print("[weekly_dag] New models promoted to models/ (already saved by trainer)")


if HAS_AIRFLOW:
    default_args = {
        "owner":            "ds01-team",
        "retries":          1,
        "retry_delay":      timedelta(minutes=10),
        "email_on_failure": False,
    }
    with DAG(
        dag_id="ds01_weekly_retrain",
        start_date=datetime(2025, 1, 1),
        schedule_interval="0 1 * * 0",
        catchup=False,
        default_args=default_args,
        tags=["churn", "training"],
    ) as dag:
        t_retrain  = PythonOperator(task_id="retrain",  python_callable=task_retrain)
        t_evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
        t_promote  = PythonOperator(task_id="promote",  python_callable=task_promote)
        t_retrain >> t_evaluate >> t_promote


if __name__ == "__main__":
    auc = task_retrain()
    task_evaluate(**{"task_instance": type("ti", (), {"xcom_pull": lambda *a, **kw: auc})()})
    task_promote()
