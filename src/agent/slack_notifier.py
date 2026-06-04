"""
src/agent/slack_notifier.py — Slack webhook notification stub for high-risk alerts.

Set SLACK_WEBHOOK_URL in .env to activate real notifications.
Falls back to console logging when the webhook is not configured.
"""

import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def send_alert(customer_id: str, risk_label: str, churn_score: float, action: str) -> bool:
    """
    Send a Slack alert for a high-risk customer.
    Returns True if the message was sent, False if it was only logged locally.
    """
    emoji = {"High": ":rotating_light:", "Medium": ":warning:", "Low": ":white_check_mark:"}.get(risk_label, ":question:")
    text = (
        f"{emoji} *Churn Alert* — Customer `{customer_id}`\n"
        f"> Risk: *{risk_label}* ({churn_score:.0%})\n"
        f"> Recommended action: `{action.upper()}`"
    )

    if not SLACK_WEBHOOK_URL:
        print(f"[slack_notifier] (no webhook configured) {text}")
        return False

    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status == 200
        print(f"[slack_notifier] Alert sent for {customer_id}: {'OK' if ok else 'FAILED'}")
        return ok
    except Exception as e:
        print(f"[slack_notifier] Send failed: {e}")
        return False
