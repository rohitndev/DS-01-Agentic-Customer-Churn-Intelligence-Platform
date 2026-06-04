"""
agent.py — LangGraph retention agent that drafts personalized messages via Groq.

Graph:
  [Analyze] → reads customer data + churn score + SHAP explanation
  [Draft]   → calls Groq LLM to write a short retention message
"""

import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"   # free-tier model on Groq


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    customer: dict
    churn_score: float
    risk_label: str
    shap_summary: str
    retention_message: str
    recommended_action: str


# ---------------------------------------------------------------------------
# Node 1: Analyze
# ---------------------------------------------------------------------------

def analyze_node(state: AgentState) -> AgentState:
    """Run prediction pipeline and build a structured explanation."""
    result = predict(state["customer"])
    shap_lines = [
        f"{i+1}. {f['feature']} (impact: {f['shap_value']:+.4f})"
        for i, f in enumerate(result["top_shap_features"])
    ]
    state["churn_score"] = result["churn_probability"]
    state["risk_label"] = result["risk_label"]
    state["shap_summary"] = "\n".join(shap_lines)
    print(f"\n[agent:analyze] Churn probability: {result['churn_probability']:.2%}")
    print(f"[agent:analyze] Risk: {result['risk_label']}")
    print(f"[agent:analyze] Top drivers:\n{state['shap_summary']}")
    return state


# ---------------------------------------------------------------------------
# Node 2: Draft
# ---------------------------------------------------------------------------

def draft_node(state: AgentState) -> AgentState:
    """Call Groq LLM to draft a retention message and recommend an action."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )

    customer = state["customer"]
    name = customer.get("customerID", "Valued Customer")
    contract = customer.get("Contract", "unknown contract")
    monthly = customer.get("MonthlyCharges", "N/A")
    tenure = customer.get("tenure", 0)

    prompt = f"""You are a customer retention specialist at a telecom company.

Customer Profile:
- ID: {name}
- Contract: {contract}
- Monthly charges: ${monthly}
- Tenure: {tenure} months
- Churn risk: {state['risk_label']} ({state['churn_score']:.0%} probability)

Top churn drivers (from ML model):
{state['shap_summary']}

Task:
1. Write a short, empathetic retention message (email or SMS, under 80 words).
   Address the top churn drivers directly.
2. On a new line, output: RECOMMENDED_ACTION: <discount|callback|escalate>
   - discount  → if the issue is pricing
   - callback  → if the issue is service quality
   - escalate  → if the risk is critical (>80%) or contract is month-to-month

Output the message first, then the recommended action on its own line."""

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.4)
    response = llm.invoke([HumanMessage(content=prompt)])
    full_text = response.content.strip()

    # Parse out the recommended action — handle plain and bold markdown formats
    action = "callback"
    message_lines = []
    for line in full_text.splitlines():
        # Strip markdown bold (**...**) and check for the tag
        clean = line.strip().lstrip("*").rstrip("*").strip()
        if clean.upper().startswith("RECOMMENDED_ACTION:"):
            raw = clean.split(":", 1)[1].strip().strip("*").strip().lower()
            # Keep only the first word (e.g. "discount" from "discount on charges")
            action = raw.split()[0] if raw else "callback"
        else:
            message_lines.append(line)

    state["retention_message"] = "\n".join(message_lines).strip()
    state["recommended_action"] = action
    return state


# ---------------------------------------------------------------------------
# Build and run the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("Analyze", analyze_node)
    builder.add_node("Draft", draft_node)
    builder.set_entry_point("Analyze")
    builder.add_edge("Analyze", "Draft")
    builder.add_edge("Draft", END)
    return builder.compile()


def run_retention_agent(customer: dict) -> dict:
    """
    Run the full LangGraph pipeline for one customer.
    Returns: { retention_message, recommended_action, churn_score, risk_label }
    """
    graph = build_graph()
    initial_state: AgentState = {
        "customer": customer,
        "churn_score": 0.0,
        "risk_label": "",
        "shap_summary": "",
        "retention_message": "",
        "recommended_action": "",
    }
    final_state = graph.invoke(initial_state)

    print("\n===== Retention Agent Output =====")
    print(f"Message:\n{final_state['retention_message']}")
    print(f"\nRecommended Action: {final_state['recommended_action'].upper()}")
    print("==================================\n")

    return {
        "retention_message": final_state["retention_message"],
        "recommended_action": final_state["recommended_action"],
        "churn_score": final_state["churn_score"],
        "risk_label": final_state["risk_label"],
    }


if __name__ == "__main__":
    sample = {
        "customerID": "DEMO-007",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55,
    }
    run_retention_agent(sample)
