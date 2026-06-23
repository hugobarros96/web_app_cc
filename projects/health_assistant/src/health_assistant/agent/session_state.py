"""Session-scoped structured state injected into the agent prompt prefix.

Storage strategy (single-tenant POC):
  - The canonical SessionContext lives in `st.session_state["session_ctx"]`
    when Streamlit is running.
  - Strands runs tool calls in WORKER THREADS where Streamlit's
    get_script_run_ctx() returns None, so reading st.session_state directly
    from a tool would miss the user's state.
  - To bridge that gap we keep a module-level reference `_GLOBAL_CTX` that
    points to the SAME SessionContext object the Streamlit thread is using.
    Worker threads read/write via the global; in-place mutations propagate
    to Streamlit's session_state because it's the same Python object.
  - Outside Streamlit (tests, CLI), only the global is used.

This is single-tenant by design - all threads in the process see one
SessionContext at a time. Multi-tenant production would key the global by
session_id and look it up at every accessor call.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CohortRef:
    name: str
    filter_code: str            # truncated to 200 chars
    row_count: int
    created_at_turn: int        # index into agent.messages when created
    sample_patient_ids: list[str]


@dataclass
class SessionContext:
    last_cohort: CohortRef | None = None
    last_patient: dict | None = None
    last_prediction: dict | None = None
    named_cohorts: dict[str, CohortRef] = field(default_factory=dict)
    web_search_enabled: bool = False


# Process-wide pointer to the active SessionContext. Worker threads see it.
_GLOBAL_CTX: SessionContext | None = None
_GLOBAL_LOCK = threading.Lock()


def _try_streamlit():
    """Return st.session_state if we are running inside a Streamlit script run,
    else None. We do NOT touch session_state outside a script run because
    Streamlit emits warnings (and in some versions raises) when accessed from
    a non-script thread."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return None
    if get_script_run_ctx(suppress_warning=True) is None:
        return None
    if not hasattr(st, "session_state"):
        return None
    return st.session_state


def get_session_ctx() -> SessionContext:
    """Return the current SessionContext, creating an empty one if absent.

    On the Streamlit thread: source of truth is st.session_state. We mirror
    the reference into _GLOBAL_CTX so worker threads can see it.
    On worker threads: returns the mirrored reference (same object as the
    Streamlit thread holds), so mutations are visible to both."""
    global _GLOBAL_CTX
    ss = _try_streamlit()
    if ss is not None:
        ctx = ss.get("session_ctx")
        if ctx is None:
            ctx = SessionContext()
            ss["session_ctx"] = ctx
        with _GLOBAL_LOCK:
            _GLOBAL_CTX = ctx
        return ctx
    # Worker thread (or non-Streamlit caller): use the global mirror.
    with _GLOBAL_LOCK:
        if _GLOBAL_CTX is None:
            _GLOBAL_CTX = SessionContext()
        return _GLOBAL_CTX


def set_session_ctx(ctx: SessionContext) -> None:
    """Replace the SessionContext. Updates both the Streamlit slot (if we are
    on the Streamlit thread) and the process-wide mirror so worker threads
    see the new reference."""
    global _GLOBAL_CTX
    ss = _try_streamlit()
    if ss is not None:
        ss["session_ctx"] = ctx
    with _GLOBAL_LOCK:
        _GLOBAL_CTX = ctx


def clear_session_ctx() -> None:
    """Wipe all session state. Used by the Streamlit clear-history button
    and by tests between cases."""
    global _GLOBAL_CTX
    fresh = SessionContext()
    ss = _try_streamlit()
    if ss is not None:
        ss["session_ctx"] = fresh
    with _GLOBAL_LOCK:
        _GLOBAL_CTX = fresh


def build_context_block(ctx: SessionContext) -> str:
    """Render the session context as the prompt-prefix block.

    Returns an empty string when the context carries nothing worth telling
    the LLM about (no cohort, no patient, no prediction, and web search off)."""
    lines: list[str] = []
    if ctx.last_cohort is not None:
        c = ctx.last_cohort
        code = c.filter_code if len(c.filter_code) <= 200 else c.filter_code[:197] + "..."
        lines.append(f'- Last cohort: "{c.name}" ({code}), {c.row_count} rows')
    if ctx.last_patient is not None:
        keys = list(ctx.last_patient.keys())[:8]
        compact = {k: ctx.last_patient[k] for k in keys}
        suffix = " (truncated)" if len(ctx.last_patient) > 8 else ""
        lines.append(
            f"- Last patient (most recent PREDICTION input only - may NOT be the "
            f"patient the user is asking about): {compact}{suffix}"
        )
    if ctx.last_prediction is not None:
        lp = ctx.last_prediction
        copd_val = lp.get("copd", {}).get("prediction") if isinstance(lp.get("copd"), dict) else lp.get("copd")
        alt_val = lp.get("alt", {}).get("prediction") if isinstance(lp.get("alt"), dict) else lp.get("alt")
        lines.append(f"- Last prediction: COPD={copd_val}, ALT={alt_val}")
    if ctx.named_cohorts:
        names = ", ".join(sorted(ctx.named_cohorts.keys()))
        lines.append(f"- Named cohorts: {names}")
    lines.append(f"- Web search: {'ON' if ctx.web_search_enabled else 'OFF'}")

    # Only show the block when there is real session-state to communicate.
    # The "Web search: OFF" default alone is not worth a block.
    real_state = any(
        x is not None for x in (ctx.last_cohort, ctx.last_patient, ctx.last_prediction)
    ) or ctx.named_cohorts or ctx.web_search_enabled

    if not real_state:
        return ""
    body = "\n".join(lines)
    return f"[session context]\n{body}\n[/session context]\n"
