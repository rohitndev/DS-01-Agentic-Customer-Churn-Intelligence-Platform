"""
drift.py — Evidently AI data drift report comparing reference vs current data.
"""

import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

sys.path.insert(0, os.path.dirname(__file__))
from ingestion import load_and_validate
from features import engineer_features, SERVICE_COLS

REPORT_DIR = "models"
SPLIT_RATIO = 0.7   # first 70% = reference, last 30% = current


def _prepare_drift_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer features and split into reference / current sets."""
    df = engineer_features(df)
    split_idx = int(len(df) * SPLIT_RATIO)
    reference = df.iloc[:split_idx].reset_index(drop=True)
    current = df.iloc[split_idx:].reset_index(drop=True)
    return reference, current


def _select_numeric_cols(df: pd.DataFrame) -> list:
    """Return numeric columns only (Evidently handles mixed types better)."""
    drop = ["customerID", "Churn"] + SERVICE_COLS
    cols = [c for c in df.select_dtypes(include="number").columns if c not in drop]
    return cols


def run_drift_report(
    data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> dict:
    """
    Run Evidently DataDriftPreset on reference vs current data.
    Returns a summary dict: { drifted_features, drift_score, total_features }.
    """
    df = load_and_validate(data_path)
    reference, current = _prepare_drift_data(df)

    numeric_cols = _select_numeric_cols(reference)
    reference_sub = reference[numeric_cols]
    current_sub = current[numeric_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_sub, current_data=current_sub)

    # Save HTML report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "drift_report.html")
    report.save_html(report_path)

    # Extract drift results from JSON representation
    result_dict = report.as_dict()
    drift_metrics = result_dict["metrics"][0]["result"]

    total = drift_metrics.get("number_of_columns", len(numeric_cols))
    drifted_count = drift_metrics.get("number_of_drifted_columns", 0)
    share = drift_metrics.get("share_of_drifted_columns", 0.0)

    # Collect per-feature drift details
    drifted_features = []
    per_col = drift_metrics.get("drift_by_columns", {})
    for col, info in per_col.items():
        if info.get("drift_detected", False):
            drifted_features.append({
                "feature": col,
                "p_value": round(info.get("p_value", 0.0), 4),
                "drift_score": round(info.get("stattest_threshold", 0.05), 4),
            })

    summary = {
        "total_features": total,
        "drifted_count": drifted_count,
        "drift_share": round(share, 4),
        "drifted_features": drifted_features,
        "report_saved_to": report_path,
    }

    print("\n===== Drift Report Summary =====")
    print(f"  Total features   : {total}")
    print(f"  Drifted features : {drifted_count} ({share:.0%})")
    if drifted_features:
        print("  Drifted columns:")
        for f in drifted_features:
            print(f"    - {f['feature']}  p={f['p_value']}")
    else:
        print("  No significant drift detected.")
    print(f"  HTML report      : {report_path}")
    print("================================\n")

    return summary


if __name__ == "__main__":
    run_drift_report()
