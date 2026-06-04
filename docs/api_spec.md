# API Specification — DS-01 Churn Intelligence API v2.0

Base URL (local): `http://127.0.0.1:8000`
Interactive docs: `http://127.0.0.1:8000/docs`

---

## GET /health
Health check.

**Response 200**
```json
{ "status": "ok" }
```

---

## POST /predict
Run ensemble churn prediction and return SHAP explanation.

**Request body** (all fields optional except none required)
```json
{
  "customerID":       "TEST-001",
  "gender":           "Female",
  "SeniorCitizen":    0,
  "Partner":          "Yes",
  "Dependents":       "No",
  "tenure":           12,
  "PhoneService":     "Yes",
  "MultipleLines":    "No",
  "InternetService":  "Fiber optic",
  "OnlineSecurity":   "No",
  "OnlineBackup":     "No",
  "DeviceProtection": "No",
  "TechSupport":      "No",
  "StreamingTV":      "Yes",
  "StreamingMovies":  "Yes",
  "Contract":         "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod":    "Electronic check",
  "MonthlyCharges":   85.50,
  "TotalCharges":     1026.0
}
```

**Response 200**
```json
{
  "churn_probability": 0.5789,
  "risk_label":        "Medium",
  "top_shap_features": [
    { "feature": "Contract_Month-to-month",      "shap_value":  0.5700 },
    { "feature": "InternetService_Fiber optic",  "shap_value":  0.3232 },
    { "feature": "PaymentMethod_Electronic check","shap_value": 0.2278 }
  ],
  "xgb_probability":  0.6204,
  "lstm_probability": 0.5167
}
```

**Errors**
- `422` — Validation error (e.g. MonthlyCharges ≤ 0)
- `503` — Models not trained yet
- `500` — Internal error

---

## POST /agent
Run LangGraph retention agent: predict → explain → draft message via Groq.

**Request body** — same schema as `/predict`

**Response 200**
```json
{
  "retention_message":  "Hi TEST-001, we've noticed...",
  "recommended_action": "discount",
  "churn_score":         0.5789,
  "risk_label":          "Medium",
  "nl_reason":           "This customer is at elevated churn risk primarily because..."
}
```

`recommended_action` values: `discount` | `callback` | `escalate`

---

## GET /drift
Run Evidently AI drift report (reference 70% / current 30% split).

**Response 200**
```json
{
  "total_features":   8,
  "drifted_count":    2,
  "drift_share":      0.25,
  "drifted_features": [
    { "feature": "MonthlyCharges", "p_value": 0.0023, "drift_score": 0.05 },
    { "feature": "tenure",         "p_value": 0.0412, "drift_score": 0.05 }
  ],
  "report_saved_to": "models/drift_report.html"
}
```
