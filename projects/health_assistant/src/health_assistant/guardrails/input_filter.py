"""Input guardrail: runs on the user's message BEFORE the agent sees it.

Order of operations:
1. Prompt-injection block - hard reject with a polite refusal.
2. PII redact - emails / phones / SSNs replaced with `[PII-REDACTED:type]`.
3. Scope filter - cosine sim vs. mean scope embedding; below threshold → refuse.

The scope check uses a MULTILINGUAL embedder (paraphrase-multilingual-MiniLM-
L12-v2) so Portuguese / Spanish / French / etc. medical questions don't get
falsely flagged as out-of-scope. The main RAG embedder (settings.embedding_model)
stays English-only since the corpus is English; this is a SEPARATE small
embedder used only for the scope check.

Each decision is logged to `artifacts/logs/guardrails.jsonl`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from health_assistant.guardrails.logger import log_decision
from health_assistant.guardrails.policies import (
    SCOPE_MIN_WORDS,
    SCOPE_QUERIES,
    SCOPE_REFUSAL,
    SCOPE_THRESHOLD,
    detect_injection,
    detect_pii,
    redact_pii,
)

INJECTION_REFUSAL = (
    "I can't help with that request. Try asking about the patient dataset, "
    "predictions, or medical knowledge."
)

# Multilingual embedder model id. ~120 MB. Trained on parallel corpora across
# 50+ languages so cosine similarity is comparable between e.g. an English
# seed query and a Portuguese user message about the same topic.
_SCOPE_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _scope_embedder():
    """Lazy import + cached load of the multilingual embedder used only for
    the scope check. Separate from settings.embedding_model (which is the
    English MiniLM used by the RAG retrievers)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=_SCOPE_EMBED_MODEL)


@lru_cache(maxsize=1)
def _scope_vector() -> np.ndarray:
    """Compute the mean embedding of the in-scope seed queries.

    Returns a unit-normalized 1D numpy array."""
    emb = _scope_embedder()
    vecs = np.asarray(emb.embed_documents(SCOPE_QUERIES))
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def _cosine_to_scope(text: str) -> float:
    emb = _scope_embedder()
    v = np.asarray(emb.embed_query(text))
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return 0.0
    v = v / v_norm
    return float(np.dot(v, _scope_vector()))


def filter_input(
    text: str,
    session_id: str = "anon",
    enable_scope_check: bool = True,
) -> dict:
    """Apply input-side guardrails.

    Returns:
        {
          "text":       text after PII redaction (empty string if blocked),
          "redactions": list of PII categories that were redacted,
          "blocked":    True if the agent should NOT see this message,
          "refusal":    str|None - text to return to the user when blocked,
        }
    """
    # 1. Injection
    if detect_injection(text):
        log_decision("input", "injection", "block", session_id=session_id)
        return {
            "text": "",
            "redactions": [],
            "blocked": True,
            "refusal": INJECTION_REFUSAL,
        }

    # 2. PII
    redactions = detect_pii(text)
    if redactions:
        log_decision("input", "pii", "redact", matches=redactions, session_id=session_id)
        text = redact_pii(text)

    # 3. Scope (multilingual). Skip for short messages (continuations).
    if enable_scope_check and len(text.split()) >= SCOPE_MIN_WORDS:
        try:
            sim = _cosine_to_scope(text)
        except Exception as e:  # don't let a failed embedding kill the request
            log_decision("input", "scope", "skipped", error=str(e), session_id=session_id)
            sim = None
        if sim is not None and sim < SCOPE_THRESHOLD:
            log_decision(
                "input", "scope", "refuse",
                cosine=float(sim), threshold=SCOPE_THRESHOLD,
                session_id=session_id,
            )
            return {
                "text": "",
                "redactions": redactions,
                "blocked": True,
                "refusal": SCOPE_REFUSAL,
            }

    log_decision("input", "all", "pass", redactions=redactions, session_id=session_id)
    return {"text": text, "redactions": redactions, "blocked": False, "refusal": None}
