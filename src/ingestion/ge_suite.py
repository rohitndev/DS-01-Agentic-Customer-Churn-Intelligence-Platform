"""
src/ingestion/ge_suite.py — Standalone Great Expectations suite definitions.
Useful for running validation outside of the pipeline (e.g., in CI checks).
"""

SUITE_NAME = "telco_suite"

EXPECTATION_DEFINITIONS = [
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "customerID",
        "description": "Customer ID must always be present",
    },
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "tenure",
        "description": "Tenure cannot be null",
    },
    {
        "expectation_type": "expect_column_values_to_be_between",
        "column": "tenure",
        "min_value": 0,
        "description": "Tenure must be non-negative",
    },
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "MonthlyCharges",
        "description": "MonthlyCharges cannot be null",
    },
    {
        "expectation_type": "expect_column_values_to_be_between",
        "column": "MonthlyCharges",
        "min_value": 0.01,
        "description": "MonthlyCharges must be positive",
    },
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "TotalCharges",
        "description": "TotalCharges cannot be null after coercion",
    },
    {
        "expectation_type": "expect_column_values_to_be_in_set",
        "column": "Churn",
        "value_set": ["Yes", "No"],
        "description": "Churn label must be binary Yes/No",
    },
]
