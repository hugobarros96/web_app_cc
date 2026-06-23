"""save_cohort - persist the most recent cohort under a user-chosen name.

Reads SessionContext.last_cohort (populated by python_analytics in Task 3.1)
and stores a copy in SessionContext.named_cohorts. Pure mutation; no disk I/O.
Session-only by design - cohorts die when the session ends.
"""
from __future__ import annotations

from dataclasses import replace

from health_assistant.agent.session_state import get_session_ctx, set_session_ctx


def save_cohort(name: str) -> dict:
    """Persist the most recent cohort under `name`.

    Returns:
        {"status": "saved", "overwrote": bool, "name": str, "row_count": int, "filter_code": str}
        {"status": "no_recent_cohort", "message": str}
        {"status": "error", "error": str}
    """
    if not isinstance(name, str):
        return {"status": "error", "error": "name must be a string"}
    name = name.strip()
    if not name:
        return {"status": "error", "error": "name must be a non-empty string"}

    ctx = get_session_ctx()
    if ctx.last_cohort is None:
        return {
            "status": "no_recent_cohort",
            "message": (
                "No cohort to save. Run an analytics query that filters the "
                "dataframe to a patient subset first (e.g. "
                "df[df.smoker == 'Yes'])."
            ),
        }

    overwrote = name in ctx.named_cohorts
    saved = replace(ctx.last_cohort, name=name)
    ctx.named_cohorts[name] = saved
    set_session_ctx(ctx)

    return {
        "status": "saved",
        "overwrote": overwrote,
        "name": name,
        "row_count": saved.row_count,
        "filter_code": saved.filter_code,
    }
