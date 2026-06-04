"""
src/explainability/nl_reason_generator.py — Convert SHAP feature list into
a human-readable natural-language churn explanation (no LLM required).
"""

# Mapping from encoded feature names to plain-English descriptions
FEATURE_EXPLANATIONS = {
    "Contract_Month-to-month":       "is on a month-to-month contract (easiest to cancel)",
    "Contract_One year":             "has a one-year contract",
    "Contract_Two year":             "is locked into a two-year contract",
    "InternetService_Fiber optic":   "uses fiber-optic internet (higher churn segment)",
    "InternetService_DSL":           "uses DSL internet",
    "PaymentMethod_Electronic check":"pays by electronic check (correlated with churn)",
    "tenure":                        "has short tenure (relatively new customer)",
    "MonthlyCharges":                "has high monthly charges",
    "CLV":                           "has low customer lifetime value",
    "charge_per_tenure":             "has high charge-to-tenure ratio (paying a lot relative to time with us)",
    "num_services":                  "uses few add-on services (low stickiness)",
    "is_high_value":                 "is not flagged as a high-value customer",
    "TotalCharges":                  "has low total spend",
    "SeniorCitizen":                 "is a senior citizen",
    "OnlineSecurity_No":             "lacks online security add-on",
    "TechSupport_No":                "lacks tech support add-on",
}


def _describe_feature(feature: str) -> str:
    for key, desc in FEATURE_EXPLANATIONS.items():
        if key.lower() in feature.lower():
            return desc
    name = feature.replace("_", " ").replace("-", " ").lower()
    return f"has elevated risk from '{name}'"


def generate_reason(shap_features: list[dict]) -> str:
    """
    Convert a list of SHAP dicts (from compute_shap) into a sentence.

    Example output:
      "This customer is high-risk primarily because they are on a
       month-to-month contract, use fiber-optic internet, and pay
       by electronic check."
    """
    if not shap_features:
        return "No significant churn drivers identified."

    drivers = [f for f in shap_features if f["direction"] == "increases_churn"][:3]
    if not drivers:
        drivers = shap_features[:3]

    descs = [_describe_feature(d["feature"]) for d in drivers]

    if len(descs) == 1:
        body = descs[0]
    elif len(descs) == 2:
        body = f"{descs[0]} and {descs[1]}"
    else:
        body = f"{descs[0]}, {descs[1]}, and {descs[2]}"

    return (
        f"This customer is at elevated churn risk primarily because "
        f"the customer {body}."
    )
