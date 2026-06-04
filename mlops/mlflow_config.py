"""
mlops/mlflow_config.py — MLflow experiment and run configuration helpers.
"""

import mlflow

EXPERIMENT_NAME = "ds01-churn"
TRACKING_URI    = "mlruns"     # local directory; swap for remote URI in prod


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> None:
    """Configure MLflow tracking URI and create experiment if needed."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] Tracking → {TRACKING_URI}  |  Experiment: {experiment}")


def get_best_run(metric: str = "val_auc") -> dict:
    """Return the run with the highest value of `metric`."""
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        return {}
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        return {}
    r = runs[0]
    return {
        "run_id":  r.info.run_id,
        "metrics": r.data.metrics,
        "params":  r.data.params,
    }
