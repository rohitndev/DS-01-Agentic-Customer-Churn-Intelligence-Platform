"""
mlops/drift_monitor.py — Evidently AI data drift report.
Splits dataset 70/30 (reference/current) and checks for feature drift.
"""

import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ingestion.pipeline import load_and_validate
from src.features.engineering import engineer_features, SERVICE_COLS

REPORT_PATH  = "models/drift_report.html"
SPLIT_RATIO  = 0.7


def _prepare(df: pd.DataFrame):
    df = engineer_features(df)
    split = int(len(df) * SPLIT_RATIO)
    ref = df.iloc[:split].reset_index(drop=True)
    cur = df.iloc[split:].reset_index(drop=True)
    drop = ["customerID", "Churn"] + SERVICE_COLS
    num_cols = [c for c in ref.select_dtypes(include="number").columns if c not in drop]
    return ref[num_cols], cur[num_cols]


def run_drift_report(data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> dict:
    df = load_and_validate(data_path)
    ref, cur = _prepare(df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    os.makedirs(os.path.dirname(REPORT_PATH) if os.path.dirname(REPORT_PATH) else ".", exist_ok=True)
    report.save_html(REPORT_PATH)

    result = report.as_dict()
    drift  = result["metrics"][0]["result"]
    total      = drift.get("number_of_columns", len(ref.columns))
    drifted_n  = drift.get("number_of_drifted_columns", 0)
    share      = drift.get("share_of_drifted_columns", 0.0)

    drifted = [
        {"feature": col, "p_value": round(info.get("p_value", 0.0), 4),
         "drift_score": round(info.get("stattest_threshold", 0.05), 4)}
        for col, info in drift.get("drift_by_columns", {}).items()
        if info.get("drift_detected", False)
    ]

    summary = {
        "total_features":   total,
        "drifted_count":    drifted_n,
        "drift_share":      round(share, 4),
        "drifted_features": drifted,
        "report_saved_to":  REPORT_PATH,
    }

    print(f"\n===== Drift Report =====")
    print(f"  Total: {total}  |  Drifted: {drifted_n} ({share:.0%})")
    for f in drifted:
        print(f"  - {f['feature']}  p={f['p_value']}")
    if not drifted:
        print("  No significant drift detected.")
    print(f"  HTML → {REPORT_PATH}\n")

    return summary


if __name__ == "__main__":
    run_drift_report()
