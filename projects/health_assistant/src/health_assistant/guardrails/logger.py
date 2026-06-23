"""Append-only JSONL logger for guardrail decisions.

Each decision row: {ts, stage, policy, decision, ...extras}. Used for demo
("here's the audit trail") and as a starting point for an MLflow attached
trace if we ever wire it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from health_assistant.config import paths

LOG_FILE = paths.logs_dir / "guardrails.jsonl"


def log_decision(stage: str, policy: str, decision: str, **extra) -> None:
    """Append one decision row to guardrails.jsonl.

    Args:
        stage: "input" or "output".
        policy: "pii" | "injection" | "scope" | "disclaimer" | "citation".
        decision: "pass" | "redact" | "block" | "inject" | "refuse" | "flag".
        **extra: free-form context (matches, session_id, etc.).
    """
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "policy": policy,
        "decision": decision,
        **extra,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
