"""Guardrail policy data: PII regex patterns, prompt-injection patterns,
scope-vector seed queries, and the clinician disclaimer.

Each policy here maps 1:1 to a Bedrock Guardrails policy in production:
  - PII regex          → Sensitive Information Filters
  - Injection patterns → Denied Topics / Custom Word Filters
  - Scope queries      → Denied Topics (out-of-scope detection)
  - Disclaimer         → Content policy (response template)
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def detect_pii(text: str) -> list[str]:
    """Return the list of PII categories detected in `text`."""
    hits: list[str] = []
    if EMAIL_RE.search(text):
        hits.append("email")
    if PHONE_RE.search(text):
        hits.append("phone")
    if SSN_RE.search(text):
        hits.append("ssn")
    return hits


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[PII-REDACTED:email]", text)
    text = PHONE_RE.sub("[PII-REDACTED:phone]", text)
    text = SSN_RE.sub("[PII-REDACTED:ssn]", text)
    return text


# ---------------------------------------------------------------------------
# Prompt-injection patterns
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |the )?previous instructions?",
    r"print (your |the )?system prompt",
    r"reveal (your |the )?(system|instructions?)",
    r"disregard (all |the )?(previous|above)",
    r"you are now",
    r"forget (all |your )?(previous|earlier)",
]


def detect_injection(text: str) -> bool:
    """True if `text` matches any known prompt-injection pattern."""
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Scope-vector seed queries
# (mean-embedded by the input filter; user queries below a cosine threshold to
# this mean are treated as out-of-scope)
# ---------------------------------------------------------------------------
SCOPE_QUERIES: list[str] = [
    "what is the predicted COPD class for this patient?",
    "how many smokers are in the dataset?",
    "compare lab results across readmitted patients",
    "what are the symptoms of diabetes?",
    "summarize the treatment plan for heart attack patients",
    "predicted alanine aminotransferase for a 60 year old male",
    "what medications was the heart attack patient taking?",
    "how many males older than 40 are readmitted?",
    "show me the BMI distribution of diabetic patients",
    "what does an elevated ALT mean?",
]

# Cosine similarity threshold below which a query is treated as out-of-scope.
# Tuned VERY permissively for POC - the scope filter exists to demo Bedrock
# Guardrails compatibility, not to be a strict gatekeeper. It still catches
# wildly off-topic queries (e.g. cooking recipes, programming questions) while
# letting through fuzzy follow-ups like "tell me more" or "what about diabetes?".
SCOPE_THRESHOLD = 0.05

# Messages shorter than this (in words) skip the scope check entirely. Short
# replies are almost always continuations of the prior turn ("use defaults",
# "yes please", "go ahead", "ok thanks") and can't be classified out of
# context.
SCOPE_MIN_WORDS = 4

SCOPE_REFUSAL = (
    "I'm a clinical analytics assistant - I can help with the patient dataset, "
    "predictive models, or medical knowledge questions."
)

# ---------------------------------------------------------------------------
# Disclaimer injected on outputs containing predictions / medical guidance
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "\n\n_This is an analytical aid, not a medical diagnosis. "
    "Decisions should be made by a qualified clinician._"
)
