"""
src/ingestion/pipeline.py — Load CSV, run Great Expectations validation,
and write Bronze / Silver Parquet layers.
"""

import os
import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

RAW_CSV   = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
BRONZE    = "data/processed/bronze/telco.parquet"
SILVER    = "data/processed/silver/telco_clean.parquet"

NON_NULL_COLS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


def load_csv(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    dropped = df["TotalCharges"].isna().sum()
    if dropped:
        print(f"[ingestion] Dropping {dropped} rows with unparseable TotalCharges")
    return df.dropna(subset=["TotalCharges"]).reset_index(drop=True)


def write_bronze(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(BRONZE), exist_ok=True)
    df.to_parquet(BRONZE, index=False)
    print(f"[ingestion] Bronze layer written -> {BRONZE}")


def write_silver(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(SILVER), exist_ok=True)
    df.to_parquet(SILVER, index=False)
    print(f"[ingestion] Silver layer written -> {SILVER}")


def validate(df: pd.DataFrame) -> bool:
    context = gx.get_context()
    datasource = context.sources.add_or_update_pandas("telco_source")
    asset = datasource.add_dataframe_asset("telco_asset")
    batch_request = asset.build_batch_request(dataframe=df)

    suite_name = "telco_suite"
    try:
        suite = context.get_expectation_suite(suite_name)
    except Exception:
        suite = context.add_expectation_suite(suite_name)

    suite.expectations = []
    for col in NON_NULL_COLS:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": col},
        ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "tenure", "min_value": 0},
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "MonthlyCharges", "min_value": 0.01},
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={"column": "Churn", "value_set": ["Yes", "No"]},
    ))
    context.save_expectation_suite(suite)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )
    results = validator.validate()

    print("\n===== Great Expectations Validation =====")
    all_pass = True
    for r in results.results:
        status = "PASS" if r.success else "FAIL"
        col = r.expectation_config.kwargs.get("column", "?")
        exp = r.expectation_config.expectation_type
        print(f"  [{status}] {col} — {exp}")
        if not r.success:
            all_pass = False
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    print("=========================================\n")
    return all_pass


def load_and_validate(path: str = RAW_CSV) -> pd.DataFrame:
    print(f"[ingestion] Loading data from {path}")
    df = load_csv(path)
    print(f"[ingestion] Loaded {len(df)} rows, {df.shape[1]} columns")
    write_bronze(df)
    validate(df)
    write_silver(df)
    return df


if __name__ == "__main__":
    df = load_and_validate()
    print(df.head())
