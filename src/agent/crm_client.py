"""
src/agent/crm_client.py — CRM system integration stub.

In production this would POST retention actions to your CRM (Salesforce,
HubSpot, etc.). For local use it logs the action to a SQLite journal.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/crm_actions.db"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retention_actions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   TEXT NOT NULL,
            action        TEXT NOT NULL,
            churn_score   REAL,
            risk_label    TEXT,
            message       TEXT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def log_retention_action(
    customer_id: str,
    action: str,
    churn_score: float,
    risk_label: str,
    message: str,
) -> int:
    """
    Persist a retention action.

    Returns the row ID of the inserted record (always >= 1).

    Production swap-in:
      Replace the SQLite block with an HTTP POST to your CRM API, e.g.:
        import requests
        requests.post(CRM_ENDPOINT, json=payload, headers={"Authorization": f"Bearer {CRM_TOKEN}"})
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        cursor = conn.execute(
            """INSERT INTO retention_actions
               (customer_id, action, churn_score, risk_label, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (customer_id, action, churn_score, risk_label, message, datetime.now(timezone.utc).isoformat()),
        )
        row_id = cursor.lastrowid or 0
    print(f"[crm_client] Logged action '{action}' for {customer_id} (row {row_id})")
    return row_id


def get_recent_actions(limit: int = 10) -> list[dict]:
    """Retrieve the most recent retention actions from the local journal."""
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM retention_actions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
