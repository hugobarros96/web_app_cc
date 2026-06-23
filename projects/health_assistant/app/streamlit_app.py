"""Data Doctor - Streamlit chat UI.

Runs the same agent that tests/test_agent_smoke.py exercises, with chat history,
inline matplotlib charts produced by `python_analytics`, and a per-turn tool
trace expander.

Run from repo root:
    streamlit run app/streamlit_app.py

In another terminal, the MLflow UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

import base64
import re
import sys
import uuid
from pathlib import Path

# Make src/ importable when streamlit is launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st  # noqa: E402

from health_assistant.agent.agent import build_agent, run  # noqa: E402
from health_assistant.agent.session_state import SessionContext  # noqa: E402
from health_assistant.attachments.image_loader import prepare_image  # noqa: E402
from health_assistant.attachments.pdf_reader import extract_text  # noqa: E402
from health_assistant.attachments.types import Attachment  # noqa: E402
from health_assistant.feedback.log import (  # noqa: E402
    eligibility_counts,
    log_feedback,
    maybe_trigger_retrain,
)

MAX_ATTACHMENTS_PER_TURN = 3
MAX_FILE_BYTES = 8 * 1024 * 1024
RETRAIN_STATUS_PATH = Path("artifacts/feedback/retrain_status.json")


# Inline chat-export "stamp" prefix: "[HH:MM, DD/MM/YYYY] Name: " or "[HH:MM] Name: ".
# We treat any such prefix as a soft "new message starts here" marker by
# replacing it with a paragraph break. Not REQUIRED for splitting - we still
# split on paragraphs / numbered lists / question marks below.
_STAMP_PREFIX_RE = re.compile(
    r"\[\d{1,2}:\d{2}(?:,\s*\d{1,2}/\d{1,2}/\d{2,4})?\]\s+[^:\n]{1,80}?:\s*",
)
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_NUMBERED_RE = re.compile(r"\n\s*\d+[.)]\s+")
# Split on "? " (question mark followed by whitespace then a capital letter,
# Latin or Latin-extended). Anchors on the next-sentence boundary so we don't
# split mid-sentence questions. The `?` itself is preserved on the previous
# chunk via a positive lookbehind below.
_Q_MARK_BOUNDARY_RE = re.compile(r"(?<=\?)\s+(?=[A-ZÀ-ſ])")


def _split_multi_question(text: str) -> list[str]:
    """Generic splitter for multi-question pastes.

    Order of attempts (first one that yields >1 part wins):
      1. Paragraph breaks (`\\n\\n`) - most reliable signal.
      2. Numbered list items at line start (`1. ... 2. ...`).
      3. Question-mark sentence boundaries (`? Next sentence...`).

    Pre-processing: chat-export stamp prefixes like `[19:48, 02/06/2026] Name:`
    are replaced with paragraph breaks so they don't survive as garbage in
    the split chunks (and so step 1 picks them up).

    Returns a list with at least one entry. Single questions return [text]
    unchanged.
    """
    text = text.strip()
    if not text:
        return [text]

    # Normalize chat-export prefixes -> paragraph breaks.
    text = _STAMP_PREFIX_RE.sub("\n\n", text).strip()

    for splitter in (_PARAGRAPH_RE, _NUMBERED_RE, _Q_MARK_BOUNDARY_RE):
        parts = [p.strip() for p in splitter.split(text) if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]

st.set_page_config(page_title="Data Doctor", page_icon="🩺", layout="wide")
st.title("🩺 Data Doctor")
st.caption(
    "Clinical-analytics assistant - predictions, dataset queries, and grounded medical answers. "
    "POC; not for clinical use."
)


# ---------------------------------------------------------------------------
# Cached agent. Keyed by session_id so each browser session owns its own
# Strands Agent instance and its own conversation memory (agent.messages).
# Two tabs with different ?sid= cannot see each other's history.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading agent (first request only)...")
def _agent_for(session_id: str):
    return build_agent()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "text": str, "figures": [b64png, ...],
    #              "tools_used": [str], "redactions": [str], "flags": [str]}
    st.session_state.history = []
# Persist session_id in the URL via query params so a browser refresh keeps
# the same MLflow session (history is wiped on hard refresh — that's expected
# — but the session_id stays consistent if the user opens this tab again).
sid_param = st.query_params.get("sid")
if sid_param:
    st.session_state.session_id = sid_param
elif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.query_params["sid"] = st.session_state.session_id
# Queue of prompts the user has submitted but the agent hasn't started yet.
# (queue[0] is currently being processed; queue[1:] are waiting.)
if "queue" not in st.session_state:
    st.session_state.queue = []
# Structured session memory (last cohort, last patient, last prediction, named
# cohorts, web-search toggle). Lives only in memory; wiped on clear-history.
if "session_ctx" not in st.session_state:
    st.session_state["session_ctx"] = SessionContext()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.queue = []
        st.session_state["session_ctx"] = SessionContext()
        # Start a fresh MLflow session
        new_sid = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_sid
        st.query_params["sid"] = new_sid
        st.rerun()

    st.markdown("---")
    ws_on = st.checkbox(
        "🌐 Enable web search",
        value=st.session_state["session_ctx"].web_search_enabled,
        help=(
            "When ON, the agent may use the web_search tool, restricted to a "
            "medical-domain allowlist (CDC, NIH, FDA, WHO, PubMed, Mayo, "
            "Cleveland Clinic, Medscape, UpToDate, NEJM, BMJ, Lancet, JAMA, "
            "etc.). Requires SERPA_API_KEY in your .env."
        ),
    )
    if ws_on != st.session_state["session_ctx"].web_search_enabled:
        st.session_state["session_ctx"].web_search_enabled = ws_on

    st.markdown("---")
    st.subheader("Try asking…")
    st.markdown(
        """
        - How many smokers are in the dataset?
        - How many males older than 40 are readmitted?
        - What medications was the heart attack patient taking?
        - What are the symptoms of seasonal allergies?
        - Predict COPD for a 55-year-old male with BMI 27.5, 3 medications, no exercise, poor diet.
        - Compare lab results across readmitted vs non-readmitted patients.
        """
    )

    st.markdown("---")
    st.subheader("Active learning")
    al_counts = eligibility_counts()
    st.markdown(
        f"**Feedback collected:** {al_counts['eligible_pending']} / "
        f"{al_counts['threshold']} (eligible)  \n"
        f"_{al_counts['total']} logged total_"
    )
    if RETRAIN_STATUS_PATH.exists():
        try:
            import json as _json

            cur = _json.loads(RETRAIN_STATUS_PATH.read_text()).get("current", {})
        except Exception:
            cur = {}
        if cur:
            st.markdown(f"**Last retrain:** `{cur.get('timestamp', 'unknown')}`")
            st.markdown(f"Status: `{cur.get('status', 'unknown')}`")
            if cur.get("delta_macro_f1") is not None:
                st.caption(f"COPD macro-F1 delta: {cur['delta_macro_f1']:+.4f}")
            if cur.get("delta_alt_mae") is not None:
                st.caption(f"ALT MAE delta: {cur['delta_alt_mae']:+.4f}")
            if cur.get("rejection_breakdown"):
                rejs = ", ".join(f"{k}={v}" for k, v in cur["rejection_breakdown"].items())
                st.caption(f"Rejected: {rejs}")

    st.markdown("---")
    st.markdown("**MLflow UI:** http://localhost:5000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        # Replay-only attachment badge (the bytes are gone; we kept names only).
        names = turn.get("attachment_names") or []
        if names:
            st.caption("📎 " + " · ".join(names))
        for fig_b64 in turn.get("figures", []) or []:
            st.image(base64.b64decode(fig_b64))
        if turn["role"] == "assistant" and turn.get("tools_used"):
            with st.expander("🛠️ Tools used"):
                for t in turn["tools_used"]:
                    st.markdown(f"- `{t}`")
        if turn.get("flags"):
            st.warning(", ".join(turn["flags"]))


def _render_feedback_widgets(turn: dict) -> None:
    """Render the active-learning feedback expander under an assistant turn
    that included a predict_patient_outcomes call.

    Reads `turn["last_prediction"]` (attached in the processing block when
    predict_patient_outcomes was used this turn). Re-keys widget keys on the
    prediction_id so each prediction in the chat history gets its own state.

    When the underlying prediction used imputed defaults, a notice tells the
    user the row will be logged for audit but NOT eligible for training (per
    the strict any-imputation rule in feedback/validation.py).
    """
    last_pred = turn.get("last_prediction")
    if not last_pred:
        return
    pid = last_pred["prediction_id"]
    imputed = last_pred.get("imputed_features") or []

    with st.expander("🩺 Was this prediction correct?", expanded=False):
        if imputed:
            st.info(
                "This prediction used imputed defaults for: "
                f"{', '.join(imputed)}. Your feedback will be logged (audit "
                "trail) but will NOT enter training data. To contribute to "
                "training, re-run the prediction with complete feature values."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Predicted COPD:** {last_pred['predicted_copd']}")
            actual_copd = st.selectbox(
                "Actual COPD class",
                options=["(no comment)", "A", "B", "C", "D"],
                key=f"fb_copd_{pid}",
            )
        with col2:
            st.markdown(f"**Predicted ALT:** {last_pred['predicted_alt']:.1f}")
            actual_alt = st.number_input(
                "Actual ALT value (mIU/L)",
                value=None,
                min_value=0.0,
                max_value=200.0,
                key=f"fb_alt_{pid}",
            )

        c_a, c_b = st.columns(2)
        with c_a:
            if st.button("👍 Both correct", key=f"fb_up_{pid}"):
                log_feedback(
                    prediction_id=pid,
                    session_id=st.session_state.session_id,
                    features=last_pred["features_used"],
                    imputed_features=imputed,
                    predicted_copd=last_pred["predicted_copd"],
                    predicted_copd_scores=last_pred["predicted_copd_scores"],
                    predicted_alt=last_pred["predicted_alt"],
                    predicted_alt_interval_80=last_pred["predicted_alt_interval_80"],
                    actual_copd=last_pred["predicted_copd"],
                    actual_alt=last_pred["predicted_alt"],
                    kind="thumbs_up",
                )
                triggered = maybe_trigger_retrain()
                if triggered:
                    st.success("Feedback logged. Retrain triggered in background.")
                else:
                    st.success("Feedback logged.")

        with c_b:
            disabled = (actual_copd == "(no comment)" and actual_alt is None)
            if st.button(
                "Submit correction", key=f"fb_dn_{pid}", disabled=disabled
            ):
                log_feedback(
                    prediction_id=pid,
                    session_id=st.session_state.session_id,
                    features=last_pred["features_used"],
                    imputed_features=imputed,
                    predicted_copd=last_pred["predicted_copd"],
                    predicted_copd_scores=last_pred["predicted_copd_scores"],
                    predicted_alt=last_pred["predicted_alt"],
                    predicted_alt_interval_80=last_pred["predicted_alt_interval_80"],
                    actual_copd=None if actual_copd == "(no comment)" else actual_copd,
                    actual_alt=actual_alt,
                    kind="correction",
                )
                triggered = maybe_trigger_retrain()
                if triggered:
                    st.success("Correction logged. Retrain triggered in background.")
                else:
                    st.success("Correction logged.")


# ---------------------------------------------------------------------------
# Replay completed turns
# ---------------------------------------------------------------------------
for turn in st.session_state.history:
    _render_turn(turn)
    if turn["role"] == "assistant":
        _render_feedback_widgets(turn)

# ---------------------------------------------------------------------------
# Chat input with inline attachments (📎). The paperclip lives inside the input
# box (always pinned with it); files attach to the message submitted with them
# (one-shot). New submissions append to the queue and trigger an immediate
# rerun, so the user sees messages stack up while the agent answers the prior
# one. Queue entries are dicts {text, attachments} so attachments travel with
# the prompt they were attached to (even if the user stacks multiple prompts).
# ---------------------------------------------------------------------------
submission = st.chat_input(
    "Ask about the patient data, predictions, or medical knowledge…",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg", "jpeg", "webp"],
    max_upload_size=MAX_FILE_BYTES // (1024 * 1024),
)
if submission:
    new_text = submission.text or ""
    uploads = list(submission.files or [])

    # Process uploads into Attachment objects. The widget already enforces
    # file_type + per-file size; these guards are a defensive backstop.
    processed_attachments: list[Attachment] = []
    if len(uploads) > MAX_ATTACHMENTS_PER_TURN:
        st.toast(
            f"Max {MAX_ATTACHMENTS_PER_TURN} attachments per turn; "
            f"using the first {MAX_ATTACHMENTS_PER_TURN}."
        )
        uploads = uploads[:MAX_ATTACHMENTS_PER_TURN]
    for up in uploads:
        if up.size > MAX_FILE_BYTES:
            st.toast(f"Skipping {up.name}: too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB).")
            continue
        data = up.getvalue()
        if up.name.lower().endswith(".pdf"):
            payload = extract_text(data)
            processed_attachments.append(Attachment(kind="pdf", name=up.name, payload=payload))
        else:
            payload = prepare_image(data)
            processed_attachments.append(Attachment(kind="image", name=up.name, payload=payload))

    # Split a pasted multi-question submit (WhatsApp export, numbered list) into
    # separate queue items. Attachments only travel with the FIRST sub-question
    # (the natural semantics: "the file applies to the next thing I asked").
    chunks = _split_multi_question(new_text) if new_text.strip() else [new_text]
    for i, chunk in enumerate(chunks):
        st.session_state.queue.append({
            "text": chunk,
            "attachments": (processed_attachments or None) if i == 0 else None,
        })
    st.rerun()

# ---------------------------------------------------------------------------
# Show queued prompts (queue[1:]) so the user sees them stacked while waiting.
# ---------------------------------------------------------------------------
if len(st.session_state.queue) > 1:
    for queued in st.session_state.queue[1:]:
        text = queued["text"] if isinstance(queued, dict) else queued
        atts = queued.get("attachments") if isinstance(queued, dict) else None
        with st.chat_message("user"):
            extra = f"  \n_({len(atts)} attachment(s))_" if atts else ""
            st.markdown(text + extra + "  \n_…queued, waiting for previous to finish._")

# ---------------------------------------------------------------------------
# Process the head of the queue (if any).
#
# CRITICAL ordering: save user + assistant turns to history IMMEDIATELY after
# the agent call returns, BEFORE any st.* widget call. Streamlit aborts an
# in-flight rerun at the next widget call when a new submission arrives — so
# anything we render AFTER the agent call risks being aborted. By writing to
# history (which is plain dict mutation, not a widget call) before rendering,
# we guarantee the answer survives an abort and is replayed on the next rerun.
# ---------------------------------------------------------------------------
if st.session_state.queue:
    current_item = st.session_state.queue[0]
    if isinstance(current_item, dict):
        current = current_item["text"]
        current_attachments = current_item.get("attachments")
    else:
        # Legacy string entries (pre-Task-5.5 reruns). Defensive fallback.
        current = current_item
        current_attachments = None

    # Show the current user message inline (this might be aborted; it's also
    # replayed from history on the next rerun, so any abort is cosmetic only)
    with st.chat_message("user"):
        st.markdown(current)
        if current_attachments:
            badges = []
            for a in current_attachments:
                if a.kind == "pdf":
                    badges.append(f"📄 {a.name}")
                else:
                    badges.append(f"🖼️ {a.name}")
            st.caption(" · ".join(badges))

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = run(
                _agent_for(st.session_state.session_id),
                current,
                history=st.session_state.history,
                session_id=st.session_state.session_id,
                attachments=current_attachments,
            )

        # If this turn used predict_patient_outcomes, snapshot the most-recent
        # prediction from SessionContext into the reply so the feedback widget
        # can render and re-render against the same prediction_id across reruns.
        ctx = st.session_state["session_ctx"]
        if (
            "predict_patient_outcomes" in (reply.get("tools_used") or [])
            and ctx.last_prediction is not None
        ):
            lp = ctx.last_prediction
            copd_block = lp.get("copd", {})
            alt_block = lp.get("alt", {})
            reply["last_prediction"] = {
                "prediction_id": lp.get("prediction_id"),
                "predicted_copd": copd_block.get("prediction"),
                "predicted_copd_scores": copd_block.get("class_scores", {}),
                "predicted_alt": float(alt_block.get("prediction", 0.0)),
                "predicted_alt_interval_80": [
                    float(x) for x in alt_block.get("interval_80", [0.0, 0.0])
                ],
                "features_used": ctx.last_patient or {},
                "imputed_features": lp.get("imputed_features", []),
            }

        # SAVE FIRST (plain dict mutation — abort-safe). Attachments are NOT
        # persisted in history (one-shot lifecycle; replay shows only the
        # text + a badge listing attachment names, not the bytes).
        history_user_turn = {"role": "user", "text": current}
        if current_attachments:
            history_user_turn["attachment_names"] = [a.name for a in current_attachments]
        st.session_state.history.append(history_user_turn)
        st.session_state.history.append({"role": "assistant", **reply})
        st.session_state.queue.pop(0)

        # Render (these are widget calls — if a new submit arrives now Streamlit
        # will abort here. That's fine: history has the answer; next rerun
        # replays it.)
        st.markdown(reply["text"])
        for fig_b64 in reply.get("figures", []) or []:
            st.image(base64.b64decode(fig_b64))
        if reply.get("tools_used"):
            with st.expander("🛠️ Tools used"):
                for t in reply["tools_used"]:
                    st.markdown(f"- `{t}`")
        if reply.get("flags"):
            st.warning(", ".join(reply["flags"]))

    # Inline feedback expander for this freshly-rendered prediction.
    _render_feedback_widgets({"role": "assistant", **reply})

    # If more items remain in the queue, kick off the next rerun.
    if st.session_state.queue:
        st.rerun()
