"""Agent tool: run pandas/matplotlib code in the restricted sandbox.

The patient dataframe `df` is preloaded once per process (lru_cache) and reused
across calls. The agent writes small snippets - counts, filters, groupbys,
charts - and gets back stdout, the last-expression value, and any matplotlib
figure as base64 PNG.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import pandas as pd

from health_assistant.agent.session_state import CohortRef, get_session_ctx, set_session_ctx
from health_assistant.analytics.sandbox import run_sandboxed
from health_assistant.config import load_patient_csv


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    return load_patient_csv()


# Per-turn figure registry. The Streamlit UI is single-user, so a module-level
# list with a lock is sufficient. Multi-user production would use contextvars
# or a per-session keyed dict.
_figures_lock = threading.Lock()
_figures_this_turn: list[str] = []


def clear_figures() -> None:
    """Reset the figure registry. Call at the start of an agent turn."""
    with _figures_lock:
        _figures_this_turn.clear()


def get_figures() -> list[str]:
    """Return a copy of figures captured this turn (base64-encoded PNGs)."""
    with _figures_lock:
        return list(_figures_this_turn)


def _maybe_record_cohort(value, executed_code: str) -> None:
    """If the executed code's final expression is a DataFrame that looks like a
    row-filter on the patient table (has patient_id, fewer rows than the full
    table, capped at 5000), persist it as SessionContext.last_cohort.

    Heuristic, not exhaustive - the agent may need to be explicit ('save this
    as X') for save_cohort to pick up an arbitrary slice."""
    if not isinstance(value, pd.DataFrame):
        return
    if "patient_id" not in value.columns:
        return
    full_n = len(_df())
    if not (0 < len(value) < min(full_n, 5000)):
        return

    ctx = get_session_ctx()
    prev_turn = ctx.last_cohort.created_at_turn if ctx.last_cohort else 0
    n = prev_turn + 1
    code = executed_code if len(executed_code) <= 200 else executed_code[:197] + "..."
    ctx.last_cohort = CohortRef(
        name=f"cohort_{n}",
        filter_code=code,
        row_count=int(len(value)),
        created_at_turn=n,
        sample_patient_ids=value["patient_id"].head(5).astype(str).tolist(),
    )
    set_session_ctx(ctx)


def python_analytics(code: str, timeout_seconds: int = 30) -> dict:
    """Execute `code` against the patient dataframe.

    Args:
        code: Python code with access to: df, pd, np, plt, sns + safe builtins.
            No `import` allowed (all libs are preloaded). End with an expression
            to return its value; use `plt.show()` to emit a chart.
        timeout_seconds: hard wall-clock limit.

    Returns:
        {"stdout": str, "value": Any, "figure_png": str|None (base64), "error": str|None}

    Side effects:
        - If a figure was produced, it is pushed onto the per-turn figure
          registry so the UI can render it alongside the agent's reply.
        - If the final expression looks like a patient-cohort filter (DataFrame
          with patient_id, fewer rows than the full table), it is recorded as
          SessionContext.last_cohort so save_cohort and follow-up turns can
          refer to it.
    """
    result = run_sandboxed(code, df=_df(), timeout_seconds=timeout_seconds)
    if result.get("figure_png"):
        with _figures_lock:
            _figures_this_turn.append(result["figure_png"])
    if not result.get("error"):
        _maybe_record_cohort(result.get("value"), code)
    # Strip the base64 PNG from what we send back to the LLM. The agent would
    # otherwise try to inline tens of thousands of tokens of image data in its
    # markdown reply and hit max_tokens. The UI reads the figure from the
    # registry (_figures_this_turn) instead.
    if "figure_png" in result and result["figure_png"]:
        result = {k: v for k, v in result.items() if k != "figure_png"}
        result["chart_rendered"] = True
    return result
