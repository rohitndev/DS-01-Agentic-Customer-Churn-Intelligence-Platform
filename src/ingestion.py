"""
ingestion.py — Load IBM Telco CSV and validate with Great Expectations.
"""

import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Columns that must not contain nulls
NON_NULL_COLS = [
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
]


def load_csv(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is read as string when it has spaces — coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Drop rows where TotalCharges could not be parsed (typically new customers)
    dropped = df["TotalCharges"].isna().sum()
    if dropped:
        print(f"[ingestion] Dropping {dropped} rows with unparseable TotalCharges")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> bool:
    """Run Great Expectations suite and print results. Returns True if all pass."""
    context = gx.get_context()

    # Build an in-memory datasource
    datasource = context.sources.add_or_update_pandas("telco_source")
    asset = datasource.add_dataframe_asset("telco_asset")
    batch_request = asset.build_batch_request(dataframe=df)

    # Create or fetch expectation suite
    suite_name = "telco_suite"
    try:
        suite = context.get_expectation_suite(suite_name)
    except Exception:
        suite = context.add_expectation_suite(suite_name)

    suite.expectations = []  # reset so re-runs don't duplicate

    # --- Expectations ---
    # 1. No nulls in key columns
    for col in NON_NULL_COLS:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col},
            )
        )

    # 2. tenure >= 0
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "tenure", "min_value": 0},
        )
    )

    # 3. MonthlyCharges > 0
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "MonthlyCharges", "min_value": 0.01},
        )
    )

    # 4. Churn column only contains "Yes" / "No"
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "Churn", "value_set": ["Yes", "No"]},
        )
    )

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

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILURES — check above'}")
    print("=========================================\n")
    return all_pass


def load_and_validate(path: str = DATA_PATH) -> pd.DataFrame:
    """Public entry point: load CSV, validate, return clean DataFrame."""
    print(f"[ingestion] Loading data from {path}")
    df = load_csv(path)
    print(f"[ingestion] Loaded {len(df)} rows, {df.shape[1]} columns")
    validate(df)
    return df


if __name__ == "__main__":
    df = load_and_validate()
    print(df.head())
    print(f"Shape: {df.shape}")
