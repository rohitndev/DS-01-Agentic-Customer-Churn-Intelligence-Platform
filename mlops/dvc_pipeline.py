"""
mlops/dvc_pipeline.py — DVC pipeline stage definitions.

In production: `dvc repro` reads dvc.yaml and runs stages in dependency order.
Here we define the same stages as Python functions for local execution and
documentation purposes.

To use DVC:
  pip install dvc
  dvc init
  dvc run -n ingest  -d data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
           -o data/processed/bronze/telco.parquet \
           "python -m src.ingestion.pipeline"
  dvc run -n features -d data/processed/silver/telco_clean.parquet \
           -o data/processed/gold/telco_features.parquet \
           "python -m src.features.engineering"
  dvc run -n train -d data/processed/gold/telco_features.parquet \
           -o models/ "python -m src.models.trainer"
"""

PIPELINE_STAGES = [
    {
        "name":    "ingest",
        "cmd":     "python -m src.ingestion.pipeline",
        "deps":    ["data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"],
        "outputs": ["data/processed/bronze/telco.parquet",
                    "data/processed/silver/telco_clean.parquet"],
    },
    {
        "name":    "features",
        "cmd":     "python -m src.features.engineering",
        "deps":    ["data/processed/silver/telco_clean.parquet"],
        "outputs": ["data/processed/gold/telco_features.parquet"],
    },
    {
        "name":    "train",
        "cmd":     "python -m src.models.trainer",
        "deps":    ["data/processed/gold/telco_features.parquet"],
        "outputs": ["models/xgb_model.joblib", "models/lstm_model.pt",
                    "models/lstm_scaler.joblib", "models/feature_cols.joblib"],
        "metrics": ["mlruns/"],
    },
    {
        "name":    "drift",
        "cmd":     "python -m mlops.drift_monitor",
        "deps":    ["data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                    "models/xgb_model.joblib"],
        "outputs": ["models/drift_report.html"],
    },
]


def print_pipeline():
    for stage in PIPELINE_STAGES:
        print(f"\nStage: {stage['name']}")
        print(f"  cmd:  {stage['cmd']}")
        print(f"  deps: {stage['deps']}")
        print(f"  out:  {stage['outputs']}")


if __name__ == "__main__":
    print_pipeline()
