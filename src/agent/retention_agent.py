"""
src/agent/retention_agent.py — LangGraph 2-node retention agent.
Node 1 (Analyze): score customer + build SHAP explanation.
Node 2 (Draft):   call Groq LLM to write a personalized retention message.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.models.predictor import predict
from src.explainability.nl_reason_generator import generate_reason

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"


class AgentState(TypedDict):
    customer:           dict
    churn_score:        float
    risk_label:         str
    shap_summary:       str
    nl_reason:          str
    retention_message:  str
    recommended_action: str


def analyze_node(state: AgentState) -> AgentState:
    result = predict(state["customer"])
    shap_lines = [
        f"{i+1}. {f['feature']} (impact: {f['shap_value']:+.4f})"
        for i, f in enumerate(result["top_shap_features"])
    ]
    state["churn_score"]  = result["churn_probability"]
    state["risk_label"]   = result["risk_label"]
    state["shap_summary"] = "\n".join(shap_lines)
    state["nl_reason"]    = generate_reason(result["top_shap_features"])
    print(f"\n[agent:analyze] Churn: {result['churn_probability']:.2%}  Risk: {result['risk_label']}")
    print(f"[agent:analyze] {state['nl_reason']}")
    return state


def draft_node(state: AgentState) -> AgentState:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Copy .env.example to .env.")

    customer = state["customer"]
    prompt = f"""You are a customer retention specialist at a telecom company.

Customer Profile:
- ID: {customer.get('customerID', 'N/A')}
- Contract: {customer.get('Contract', 'N/A')}
- Monthly charges: ${customer.get('MonthlyCharges', 'N/A')}
- Tenure: {customer.get('tenure', 0)} months
- Churn risk: {state['risk_label']} ({state['churn_score']:.0%} probability)

ML Explanation:
{state['nl_reason']}

Top churn drivers:
{state['shap_summary']}

Task:
1. Write a short, empathetic retention message (email or SMS, under 80 words).
2. On its own line write: RECOMMENDED_ACTION: <discount|callback|escalate>
   - discount  → pricing concern
   - callback  → service quality concern
   - escalate  → critical risk (>80%) or month-to-month contract"""

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.4)
    response = llm.invoke([HumanMessage(content=prompt)])
    full_text = response.content.strip()

    action = "callback"
    message_lines = []
    for line in full_text.splitlines():
        clean = line.strip().lstrip("*").rstrip("*").strip()
        if clean.upper().startswith("RECOMMENDED_ACTION:"):
            raw = clean.split(":", 1)[1].strip().strip("*").strip().lower()
            action = raw.split()[0] if raw else "callback"
        else:
            message_lines.append(line)

    state["retention_message"]  = "\n".join(message_lines).strip()
    state["recommended_action"] = action
    return state


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("Analyze", analyze_node)
    builder.add_node("Draft",   draft_node)
    builder.set_entry_point("Analyze")
    builder.add_edge("Analyze", "Draft")
    builder.add_edge("Draft",   END)
    return builder.compile()


def run_retention_agent(customer: dict) -> dict:
    graph = build_graph()
    final = graph.invoke({
        "customer":           customer,
        "churn_score":        0.0,
        "risk_label":         "",
        "shap_summary":       "",
        "nl_reason":          "",
        "retention_message":  "",
        "recommended_action": "",
    })
    print(f"\n===== Retention Agent =====")
    print(f"Message:\n{final['retention_message']}")
    print(f"\nAction: {final['recommended_action'].upper()}")
    print(f"===========================\n")
    return {
        "retention_message":  final["retention_message"],
        "recommended_action": final["recommended_action"],
        "churn_score":        final["churn_score"],
        "risk_label":         final["risk_label"],
        "nl_reason":          final["nl_reason"],
    }


if __name__ == "__main__":
    sample = {
        "customerID": "DEMO-007", "gender": "Male", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 79.85, "TotalCharges": 239.55,
    }
    run_retention_agent(sample)
