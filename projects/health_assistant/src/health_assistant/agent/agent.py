"""Strands agent factory + the single `run(message, history)` entry point used by
the Streamlit UI.

The agent wires together: model provider, system prompt, 4 typed tools, input/
output guardrails, and MLflow tracing.
"""
from __future__ import annotations

import re
import threading
from typing import Any

import mlflow
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.handlers.callback_handler import null_callback_handler
from strands.types.exceptions import ConcurrencyException

from health_assistant.agent.model_provider import get_model
from health_assistant.agent.session_state import build_context_block, get_session_ctx
from health_assistant.agent.system_prompt import build_system_prompt
from health_assistant.attachments.types import Attachment
from health_assistant.guardrails.input_filter import filter_input
from health_assistant.guardrails.output_filter import filter_output
from health_assistant.observability.mlflow_setup import setup_mlflow
from health_assistant.tools.compare_patients import compare_patients as _compare_patients
from health_assistant.tools.predict_patient_outcomes import (
    predict_patient_outcomes as _predict,
)
from health_assistant.tools.python_analytics import (
    clear_figures as _clear_figures,
    get_figures as _get_figures,
    python_analytics as _analytics,
)
from health_assistant.tools.save_cohort import save_cohort as _save_cohort
from health_assistant.tools.search_clinical_documents import (
    search_clinical_documents as _search_clinical,
)
from health_assistant.tools.search_medical_knowledge import (
    search_medical_knowledge as _search_medical,
)
from health_assistant.tools.web_search import web_search as _web_search


# ---------------------------------------------------------------------------
# Tool wrappers (Strands @tool decorator wraps the underlying functions; we
# also decorate each with @mlflow.trace so calls show up in MLflow traces).
# ---------------------------------------------------------------------------
@tool
@mlflow.trace
def predict_patient_outcomes(
    features: dict,
    ask_back: bool = True,
    assumptions: list | None = None,
) -> dict:
    """Predict COPD GOLD class AND ALT value for ONE hypothetical / described patient.

    WHEN TO CALL:
    - User describes a patient and asks for a prediction
      ("predicted COPD for a 55yo male with BMI 27.5, 3 medications, no exercise")
    - User says "predict", "predicted", "what would the model say for…",
      "expected COPD class", "expected ALT".

    PROTOCOL:
    - Call with ask_back=True first. If status='needs_input', surface the
      missing features (most important first) and wait for either (a) more
      values or (b) "go ahead" / "use defaults" before re-calling.
    - Map NL descriptors via `assumptions` (e.g. "athlete" -> exercise_frequency=High,
      "lives downtown" -> urban=1).
    - Always report BOTH COPD class AND ALT value, even if the user asked for
      one. class_scores are RAW softmax outputs, NOT calibrated probabilities -
      say "the model ranks class C highest (score X)", NEVER "X% chance".

    NEVER use this tool for:
    - Counting / aggregating over the dataset -> python_analytics
    - Looking up records in the markdown corpus -> search_clinical_documents
    - General medical knowledge / definitions -> search_medical_knowledge
    """
    return _predict(features=features, ask_back=ask_back, assumptions=assumptions)


@tool
@mlflow.trace
def python_analytics(code: str) -> dict:
    """Run pandas / matplotlib code over the 10k-row patient dataframe `df` ONLY.

    WHEN TO CALL:
    - Counts and aggregates ("how many smokers?", "average BMI by diagnosis_code")
    - Filters and cohorts ("males over 40 readmitted")
    - Distributions and charts ("histogram of age", "boxplot of ALT by readmitted")
    - Group comparisons ("compare lab results across readmitted vs not")

    HOW TO WRITE THE SNIPPET:
    - `df`, pd, np, plt, sns are preloaded; no imports allowed.
    - End with an expression so it becomes the returned `value`.
    - Use plt.show() to emit a chart (it streams to the UI; do NOT inline bytes).
    - For multi-group / multi-lab comparisons, compute EVERY requested column in
      a SINGLE groupby/agg call and end with `.to_markdown()` so the caller can
      paste the table through. Never report a requested stat as "not computed".

    ON ERROR (mandatory retry, never surface a raw tool error):
    - If the response has a non-null `error`, you MUST call this tool AGAIN
      with a simpler snippet before replying to the user. Common rewrites:
        * ImportError / "Missing optional dependency" -> drop the offending call
          (e.g. replace `.to_markdown()` with `.to_string()` or `print(df)`)
        * "name not allowed" / "import not allowed" -> use only df/pd/np/plt/sns
        * SyntaxError -> simplify; split into smaller steps
    - Only after one retry still fails, tell the user in plain language what
      you tried and give the answer as a plain-text table or paragraph from
      the numbers you DO have. Never paste the raw Python traceback.

    NEVER use this tool for:
    - Predictions for a described patient -> predict_patient_outcomes
    - Facts about specific clinical records (markdown corpus) -> search_clinical_documents
    - General medical knowledge / definitions -> search_medical_knowledge

    The dataframe holds tabular features and two model targets. It does NOT
    contain free-text records or medical-knowledge content.
    """
    return _analytics(code=code)


@tool
@mlflow.trace
def search_clinical_documents(query: str, k: int = 5) -> list[dict]:
    """Search the 1,050 patient-encounter MARKDOWN RECORDS (lab reports,
    discharge summaries, consultation notes, radiology reports, referral
    letters, prescriptions).

    WHEN TO CALL: only when the user explicitly points at the corpus.
    - "look up record P00042 in the documents"
    - "summarize the treatment plan in the diabetic discharge summaries"
    - "what does the lab report for case X say"
    - "show me an example from the corpus"

    CRITICAL - do NOT use this tool when:
    - The user says "the patient I told you about" / "that patient" /
      "the 55yo male" / "the heart attack patient (we just predicted)".
      Those refer to a HYPOTHETICAL patient described EARLIER in this chat.
      Answer from the conversation history + the [session context] block
      (Last patient = the feature dict from the most recent prediction).
      Free-text details the user mentioned (drug names like "Zoloft",
      symptoms) are in the conversation history - scan for them.
    - The question is about dataset-wide counts -> python_analytics.
    - The question is general medical knowledge -> search_medical_knowledge.

    Returns top-k hybrid (FAISS + BM25 + RRF + reranker) hits. Cite each
    by `source_file` and section heading.
    """
    return _search_clinical(query=query, k=k)


@tool
@mlflow.trace
def search_medical_knowledge(query: str, k: int = 5) -> list[dict]:
    """Search medical textbooks (MedRAG: Harrison's Internal Medicine, Robbins
    Pathology, Nelson Pediatrics, etc.) for GENERAL medical knowledge.

    WHEN TO CALL: definitions, symptoms, mechanisms, standard-of-care, lab
    reference ranges, "what is X", "how is X treated", "side effects of X".
    Always prefer this over web_search for textbook knowledge.

    NEVER use this tool for:
    - Patient-population stats -> python_analytics
    - Specific clinical records -> search_clinical_documents
    - Predictions -> predict_patient_outcomes

    Returns top-k hybrid (FAISS + BM25 + RRF + reranker) hits. Cite each
    by textbook title.
    """
    return _search_medical(query=query, k=k)


@tool
@mlflow.trace
def compare_patients(patient_ids: list[str], include_predictions: bool = True) -> dict:
    """Side-by-side compare 2 to 5 patients BY patient_id (P00000, P00001, ...)
    in ONE call.

    WHEN TO CALL:
    - "compare P00042 and P00115"
    - "show me the differences between these patients"
    - "compare these 3 readmitted patients" (after python_analytics found them)

    Returns a feature table, COPD + ALT predictions per patient, and a
    bar-chart PNG (auto-rendered inline by the UI). The tool already handles
    predictions for all patients - do NOT loop predict_patient_outcomes.
    Render the returned `table` (list of dicts) as a markdown table; reference
    the chart in words (chart_rendered=true means it's shown in the UI).

    NEVER use this tool for:
    - Hypothetical / described patients (no patient_id) -> predict_patient_outcomes
    - Population-level comparisons -> python_analytics with groupby
    - More than 5 patients (use python_analytics for cohort stats instead)
    """
    return _compare_patients(patient_ids=patient_ids, include_predictions=include_predictions)


@tool
@mlflow.trace
def save_cohort(name: str) -> dict:
    """Name + remember the MOST RECENT python_analytics cohort.

    WHEN TO CALL:
    - "save this as smokers_over_40"
    - "remember these patients as cohort_A"
    - "call them the high-risk group"

    Reads the most recent cohort from session state (populated by
    python_analytics whenever its final expression is a DataFrame filter on
    the patient table). Returns status=no_recent_cohort if the user has not
    yet run a qualifying filter - when that happens, tell them so and
    suggest running an analytics query first.

    Cohorts are session-only by design - do NOT promise to persist them
    across sessions.
    """
    return _save_cohort(name=name)


@tool
@mlflow.trace
def web_search(query: str, k: int = 5, sites: list[str] | None = None) -> dict:
    """Use ONLY when the question needs CURRENT or RECENT medical information
    that the textbook corpus (search_medical_knowledge) might not cover -
    e.g. latest guidelines, drug recalls, post-2023 changes to standard of care.

    Examples:
    - "What are the 2024 GOLD COPD guideline updates?"
    - "Has metformin been recalled recently?"
    - "Latest BNP cutoffs for heart failure diagnosis"

    Pick `sites` from this allowlist - the 5 to 10 domains most relevant to
    the question (consider the user's language and topic). Unknown / off-list
    sites are silently dropped.

    ENGLISH GENERAL / GOVERNMENT:
      cdc.gov, nih.gov, fda.gov, medlineplus.gov, who.int, escardio.org
    ACADEMIC / INDEX:
      pubmed.ncbi.nlm.nih.gov, ncbi.nlm.nih.gov
    SECONDARY SOURCES:
      uptodate.com, merckmanuals.com, msdmanuals.com
    JOURNALS / PROFESSIONAL BODIES:
      ahajournals.org, diabetesjournals.org, thelancet.com,
      nejm.org, bmj.com, jamanetwork.com
    EUROPEAN INSTITUTIONS:
      ema.europa.eu, ecdc.europa.eu
    PORTUGUESE HEALTH SYSTEM + SOCIETIES:
      dgs.pt, sns.gov.pt, sns24.gov.pt, infarmed.pt, spms.min-saude.pt,
      ordemdosmedicos.pt, spginecologia.pt, saudereprodutiva.dgs.pt

    Examples of site picks:
    - "2024 GOLD COPD updates" -> sites=["nih.gov", "cdc.gov", "nejm.org",
        "thelancet.com", "pubmed.ncbi.nlm.nih.gov", "escardio.org"]
    - "rastreio cancro colo do utero portugal" -> sites=["dgs.pt",
        "sns.gov.pt", "spginecologia.pt", "saudereprodutiva.dgs.pt",
        "pubmed.ncbi.nlm.nih.gov", "who.int"]

    Returns top-k hits PLUS `sites_used` (the ones we biased to),
    `sites_dropped` (anything you passed that wasn't on the allowlist), and
    `filtered_out_domains` (raw Serper hits the post-filter cut). CITE BY
    DOMAIN in your reply (e.g. "Per Mayo Clinic, ...") and include the link.

    DO NOT use this tool for:
    - General textbook medical knowledge -> search_medical_knowledge
    - Anything in the patient dataset -> python_analytics / search_clinical_documents
    - Predictions -> predict_patient_outcomes
    - When the session-context block shows "Web search: OFF" - the tool will
      return status=disabled. Do NOT retry. Answer with a knowledge-cutoff
      disclaimer instead.
    - Retrying the same query multiple times when status=no_allowed_results.
      One call per question is enough; report what was found and what wasn't.
    """
    return _web_search(query=query, k=k, sites=sites)


# ---------------------------------------------------------------------------
# Factory + run()
# ---------------------------------------------------------------------------
def _warm_models() -> None:
    """Pre-load the three HuggingFace models the search/RAG tools use so the
    first user query doesn't see model-loading progress bars mid-conversation.

    Each model is wrapped in its own @lru_cache, so calling these getters
    forces a one-time load that stays cached for the process lifetime.
    Failures are tolerated (we just log and continue - the lazy path still
    works on the first actual query)."""
    try:
        from health_assistant.rag.reranker import get_reranker
        from health_assistant.tools.search_clinical_documents import (
            _retriever as _clinical_retriever,
        )
        from health_assistant.tools.search_medical_knowledge import (
            _retriever as _medical_retriever,
        )

        _clinical_retriever()
        _medical_retriever()
        get_reranker()
    except Exception as e:  # pragma: no cover - best-effort warmup
        print(f"[warm_models] skipped (will lazy-load on first use): {e}")


def build_agent() -> Agent:
    """Create a fresh Strands agent. Caller should reuse across turns.

    All seven tools are registered up front; web_search is then added/removed
    per turn to match the sidebar toggle via `set_web_search_enabled` (Strands
    rebuilds the tool list from the registry each turn, so no agent rebuild is
    needed).

    Side effect: pre-loads the HuggingFace retriever embedders and the
    cross-encoder reranker so the first user query doesn't see model-loading
    output. Adds ~5-10s to startup; cuts ~5-10s off the first user turn."""
    setup_mlflow()
    _warm_models()
    return Agent(
        model=get_model(),
        system_prompt=build_system_prompt(),
        # Suppress Strands' default PrintingCallbackHandler. It writes tokens
        # to stdout as they stream, which (a) spams the terminal and (b)
        # crashes with "I/O operation on closed file" when Streamlit aborts
        # an in-flight rerun while the worker thread is mid-stream.
        callback_handler=null_callback_handler,
        # Keep ~40 messages of history and truncate stale tool results. We tried
        # 20 to reduce context confusion, but the real culprits there were stale
        # [session context] blocks (now scrubbed each turn) and a broken
        # tools_used diff (now identity-based) - NOT the window. 20 was too tight:
        # an early free-text patient description could be trimmed out before a
        # follow-up question (e.g. "what meds was the heart attack patient on?").
        # 40 keeps a few turns of context alive; switch to a SummarizingConversation
        # Manager if even longer recall is needed.
        conversation_manager=SlidingWindowConversationManager(
            window_size=40, should_truncate_results=True
        ),
        tools=[
            predict_patient_outcomes,
            python_analytics,
            search_clinical_documents,
            search_medical_knowledge,
            compare_patients,
            save_cohort,
            web_search,
        ],
    )


_SESSION_CTX_RE = re.compile(r"\[session context\].*?\[/session context\]\n?", re.DOTALL)


def _strip_session_context_from_history(agent: Agent) -> None:
    """Remove the `[session context]` block from PRIOR stored user messages.

    The block is re-prepended fresh to each turn's user message, so leaving it
    on past turns makes history accumulate many stale, mutually-contradictory
    snapshots (old "Last prediction" vs the new one) that confuse the model on
    the most recent question. State is canonical in SessionContext, so we keep
    the block only on the current turn and scrub it everywhere else. Safe to
    run every turn; it only touches user text blocks, never tool-use pairs."""
    for msg in agent.messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and "[session context]" in block["text"]
            ):
                block["text"] = _SESSION_CTX_RE.sub("", block["text"])


def set_web_search_enabled(agent: Agent, enabled: bool) -> None:
    """Add or remove the web_search tool on an existing agent so its available
    tools match the sidebar toggle - without rebuilding the agent.

    Strands' ToolRegistry.get_all_tools_config() rebuilds the tool list from
    `registry.registry` on every turn, so adding/removing here takes effect on
    the next invocation while leaving the agent's conversation memory, traces,
    and identity untouched.

    Strands exposes `register_tool` to add but no public 'unregister', so removal
    pops the registry dict directly. The key is the tool's name ('web_search').
    Both directions are idempotent. Covered by tests/test_web_search_toggle.py."""
    reg = agent.tool_registry.registry
    if enabled and "web_search" not in reg:
        agent.tool_registry.register_tool(web_search)
    elif not enabled and "web_search" in reg:
        del reg["web_search"]


def _extract_text(result: Any) -> str:
    """Pull the text out of an AgentResult's final assistant Message."""
    msg = getattr(result, "message", None)
    if msg is None:
        return str(result)
    if isinstance(msg, dict):
        content = msg.get("content", [])
    else:
        content = getattr(msg, "content", [])
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(block["text"])
        else:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def _extract_tools_used_from_messages(agent: Agent, before_ids: set[int]) -> list[str]:
    """Return tool names used in the messages ADDED this turn.

    We diff `agent.messages` (rather than Strands' `result.metrics.tool_metrics`,
    which is accumulated across the agent's whole lifetime and conflates earlier
    turns). Crucially we diff by message object IDENTITY, not by absolute index:
    the conversation manager (SlidingWindowConversationManager.apply_management)
    trims the OLDEST messages from the front of `agent.messages` in place during
    the turn, so an index slice like `messages[len_before:]` would silently miss
    this turn's tool calls once trimming kicks in - dropping tools_used to [] and
    with it the disclaimer, the Tools-used panel, AND the feedback widget."""
    used: list[str] = []
    for msg in agent.messages:
        if id(msg) in before_ids:
            continue  # existed before this turn (survived trimming)
        content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
        for block in content or []:
            tu = None
            if isinstance(block, dict):
                tu = block.get("toolUse")
            else:
                tu = getattr(block, "toolUse", None)
            if tu:
                name = tu.get("name") if isinstance(tu, dict) else getattr(tu, "name", None)
                if name and name not in used:
                    used.append(name)
    return used


# Serialize all agent invocations across reruns. Streamlit normally does this
# already (one script thread per session), but Strands' background-worker model
# can keep an old invocation's lock "in flight" while a new rerun starts the
# next call, triggering ConcurrencyException. The explicit lock guarantees one
# in-flight call at a time, regardless of how Streamlit schedules reruns.
_AGENT_LOCK = threading.Lock()


@mlflow.trace(name="chat_turn")
def _invoke_agent(agent: Agent, prompt, session_id: str):
    """Wrap one agent turn in a parent trace tagged with the session_id.

    MLflow groups traces by `session_id` under the experiment's "Chat Sessions"
    view; the `user` field is also surfaced in the sidebar. All tool-call
    sub-traces (created by the @mlflow.trace decorators on each tool function)
    become children of this parent trace and inherit the session linkage.

    `prompt` is either a plain string (no attachments) or a list of content
    blocks (Strands routes the list through to the underlying provider as a
    multi-content user message)."""
    mlflow.update_current_trace(session_id=session_id, user="streamlit-user")
    with _AGENT_LOCK:
        return agent(prompt)


def _build_agent_input(
    prompt: str,
    attachments: list[Attachment] | None,
    session_context_block: str,
):
    """Combine the session-context block, attachments, and user prompt into the
    shape Strands expects. Returns a plain string when no attachments are
    present (so v1 behavior is preserved); otherwise a content-block list.

    When attachments are present, the order is:
        [session_context_block (if any), *attachment_blocks, user_prompt]
    """
    if not attachments:
        return (session_context_block + prompt) if session_context_block else prompt

    blocks: list = []
    if session_context_block:
        blocks.append({"text": session_context_block.rstrip()})
    for att in attachments:
        if att.kind == "pdf":
            p = att.payload
            header = f"[Attached PDF: {att.name}, {p['page_count']} pages"
            if p.get("truncated"):
                header += ", text truncated to first 8K tokens"
            if p.get("scanned_warning"):
                header += "; appears scanned, OCR not applied"
            header += "]"
            blocks.append({"text": f"{header}\n{p['text']}\n[End PDF]"})
        elif att.kind == "image":
            p = att.payload
            blocks.append({"image": {"format": p["format"],
                                     "source": {"bytes": p["bytes"]}}})
    blocks.append({"text": prompt})
    return blocks


def run(
    agent: Agent,
    message: str,
    history: list[dict] | None = None,  # accepted for UI compatibility; Agent owns memory
    session_id: str = "anon",
    attachments: list[Attachment] | None = None,
) -> dict:
    """Run one turn through the full pipeline: input guardrail → agent → output guardrail.

    Returns:
        {"text": str, "tools_used": list[str], "redactions": list[str],
         "flags": list[str], "figures": list[str]}
    """
    # Input guardrail
    pre = filter_input(message, session_id=session_id)
    if pre["blocked"]:
        return {
            "text": pre["refusal"],
            "tools_used": [],
            "redactions": pre["redactions"],
            "flags": ["blocked"],
            "figures": [],
        }

    # Reset per-turn figure registry so we only collect this turn's charts.
    _clear_figures()

    # Sync the web_search tool to the sidebar toggle: register it when on,
    # remove it when off, so the model literally cannot call it while disabled
    # (the in-tool disabled-guard remains as defense in depth). Done in place -
    # no agent rebuild.
    ctx = get_session_ctx()
    set_web_search_enabled(agent, ctx.web_search_enabled)

    # Scrub stale [session context] blocks from prior turns so only THIS turn's
    # (freshly built below) carries current state - prevents old snapshots from
    # contradicting the latest one and confusing the model on recent questions.
    _strip_session_context_from_history(agent)

    # Build session-context prefix + attachment content blocks. When no
    # attachments are present and the context block is empty, the input is
    # just the user prompt - same shape as v1.
    ctx_block = build_context_block(ctx)
    agent_input = _build_agent_input(pre["text"], attachments, ctx_block)

    # Snapshot the IDENTITIES of existing messages BEFORE invocation so we can
    # compute tools used in just this turn. Identity (not index) because the
    # conversation manager trims old messages from the front during the turn -
    # see _extract_tools_used_from_messages.
    msg_ids_before = {id(m) for m in agent.messages}

    # Agent invocation under a session-tagged parent trace.
    # Self-healing: if a prior call left the agent's internal lock set (very
    # rare; happens when an aborted Streamlit rerun killed the worker mid-
    # flight), rebuild a fresh agent and retry once.
    try:
        result = _invoke_agent(agent, agent_input, session_id)
    except ConcurrencyException:
        agent = build_agent()
        set_web_search_enabled(agent, get_session_ctx().web_search_enabled)
        msg_ids_before = {id(m) for m in agent.messages}
        result = _invoke_agent(agent, agent_input, session_id)

    text = _extract_text(result)
    tools_used = _extract_tools_used_from_messages(agent, msg_ids_before)
    figures = _get_figures()

    # Output guardrail. image_attached gates the image-interpretation check
    # so legitimate prediction explanations ("consistent with the COPD
    # diagnosis...") don't false-flag.
    image_attached = bool(attachments) and any(a.kind == "image" for a in attachments)
    post = filter_output(
        text, tools_used=tools_used, session_id=session_id, image_attached=image_attached
    )

    return {
        "text": post["text"],
        "tools_used": tools_used,
        "redactions": pre["redactions"] + post["redactions"],
        "flags": post["flags"],
        "figures": figures,
    }
