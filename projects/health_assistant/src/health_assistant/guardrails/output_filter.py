"""Output guardrail: runs on the agent's response BEFORE the user sees it.

Two concerns (intentionally trimmed for POC):
1. PII rescan - catch anything the agent may have echoed back.
2. Disclaimer injection - append the clinician disclaimer when the response is
   prediction- or medical-advice-flavored (any RAG/predict/web tool was used).

Previously this module also emitted a `missing_citation` warning and an
`image_interpretation_attempt` warning. Both were non-blocking flags that
surfaced in the UI but produced too many false positives (the LLM's natural
phrasing for clinical explanations contains "consistent with", "diagnos-",
etc.), so they were removed. The clinical-safety story relies on the
disclaimer injection plus the system-prompt rules, not on these regex flags.
"""
from __future__ import annotations

from health_assistant.guardrails.logger import log_decision
from health_assistant.guardrails.policies import DISCLAIMER, detect_pii, redact_pii

# Tools whose responses should always carry the medical disclaimer.
PREDICTION_TRIGGERS = {
    "predict_patient_outcomes",
    "search_clinical_documents",
    "search_medical_knowledge",
    "web_search",
}


def filter_output(
    text: str,
    tools_used: list[str] | None = None,
    session_id: str = "anon",
    image_attached: bool = False,  # kept for caller compatibility; unused
) -> dict:
    """Apply output-side guardrails.

    Returns:
        {"text": redacted/decorated text,
         "redactions": list of PII categories caught on the output,
         "flags": list of non-blocking concerns (currently always empty;
                  kept in the return shape for caller compatibility)}
    """
    tools_used = tools_used or []
    out_text = text
    flags: list[str] = []

    # 1. PII rescan
    leaks = detect_pii(out_text)
    if leaks:
        log_decision("output", "pii", "redact", matches=leaks, session_id=session_id)
        out_text = redact_pii(out_text)

    # 2. Disclaimer
    if any(t in PREDICTION_TRIGGERS for t in tools_used):
        if "analytical aid" not in out_text.lower():
            out_text = out_text.rstrip() + DISCLAIMER
            log_decision("output", "disclaimer", "inject", session_id=session_id)

    return {"text": out_text, "redactions": leaks, "flags": flags}
