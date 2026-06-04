"""
src/features/feature_store.py — Hopsworks Feature Store integration stub.

In production this would push the Gold feature matrix to a Hopsworks
feature group so downstream models and serving always read from a
versioned, consistent feature store.

For local/college use this writes to the Gold Parquet layer instead.
"""

import os
import pandas as pd

GOLD = "data/processed/gold/telco_features.parquet"

FEATURE_GROUP_NAME    = "telco_churn_features"
FEATURE_GROUP_VERSION = 1


def push_to_feature_store(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Push engineered features to Hopsworks (production) or Gold Parquet (local).

    To enable Hopsworks:
      1. pip install hopsworks
      2. Set HOPSWORKS_API_KEY env var
      3. Uncomment the hopsworks block below
    """
    # --- Hopsworks (production path) ---
    # import hopsworks
    # project = hopsworks.login()
    # fs = project.get_feature_store()
    # fg = fs.get_or_create_feature_group(
    #     name=FEATURE_GROUP_NAME,
    #     version=FEATURE_GROUP_VERSION,
    #     primary_key=["customerID"],
    #     description="Telco churn engineered features",
    # )
    # combined = X.copy(); combined["Churn"] = y
    # fg.insert(combined)
    # print(f"[feature_store] Pushed {len(combined)} rows to Hopsworks FG")

    # --- Local path (Gold Parquet) ---
    os.makedirs(os.path.dirname(GOLD), exist_ok=True)
    combined = X.copy()
    combined["Churn"] = y
    combined.to_parquet(GOLD, index=False)
    print(f"[feature_store] Gold Parquet written -> {GOLD}  ({len(combined)} rows)")


def read_from_feature_store() -> tuple[pd.DataFrame, pd.Series]:
    """Read features from Gold Parquet (or Hopsworks in production)."""
    if not os.path.exists(GOLD):
        raise FileNotFoundError(
            f"Gold layer not found at {GOLD}. Run the feature engineering pipeline first."
        )
    df = pd.read_parquet(GOLD)
    y = df.pop("Churn")
    return df, y
