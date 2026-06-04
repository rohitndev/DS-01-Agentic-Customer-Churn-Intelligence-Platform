"""
src/explainability/shap_explainer.py — SHAP value computation for XGBoost.
Returns per-feature attribution for any scored customer.
"""

import shap
import pandas as pd


def compute_shap(xgb_model, X: pd.DataFrame) -> list[dict]:
    """
    Compute SHAP values for X using a trained XGBoost model.
    Returns the top-N features sorted by absolute SHAP impact.
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X)
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

    feature_cols = list(X.columns)
    series = pd.Series(sv, index=feature_cols).abs().sort_values(ascending=False)

    return [
        {
            "feature":    feat,
            "shap_value": round(float(sv[feature_cols.index(feat)]), 4),
            "direction":  "increases_churn" if sv[feature_cols.index(feat)] > 0 else "decreases_churn",
        }
        for feat in series.index
    ]


def top_n_shap(xgb_model, X: pd.DataFrame, n: int = 3) -> list[dict]:
    """Return only the top-N SHAP features."""
    return compute_shap(xgb_model, X)[:n]
