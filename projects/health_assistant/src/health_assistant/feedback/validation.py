"""Feedback row validation gate: schema, any-imputation, duplicate detection.

Used by retrain.py to decide which feedback rows enter the next
training run. The eligibility flag set at write-time (feedback/log.py) is the
fast path - this module is the belt-and-suspenders check before training.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from health_assistant.models.feature_schema import FEATURE_NAMES, FEATURE_SPEC

_COPD_VALID = {"A", "B", "C", "D"}
_ALT_RANGE = (0.0, 200.0)

# FEATURE_SPEC does not encode numeric ranges; we keep clinically sensible
# bounds here so the validator can reject obviously-bad feedback rows
# (negative age, BMI of 500, etc.) without trusting whatever the user typed.
_NUMERIC_RANGES = {
    "age": (0, 120),
    "bmi": (10.0, 60.0),
    "medication_count": (0, 30),
    "days_hospitalized": (0, 365),
    "last_lab_glucose": (30.0, 500.0),
    "albumin_globulin_ratio": (0.1, 3.0),
}


def feature_hash(features: dict) -> str:
    """Stable hash of a feature dict.

    Normalizes floats to 1 decimal and strings to stripped-lowercased so trivial
    formatting differences don't bypass duplicate detection."""
    norm: dict = {}
    for name in sorted(features.keys()):
        v = features[name]
        if isinstance(v, float):
            v = round(v, 1)
        elif isinstance(v, str):
            v = v.strip().lower()
        norm[name] = v
    h = hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()
    return h[:16]


def _check_schema(row: dict) -> str | None:
    feats = row.get("features", {})
    for name in FEATURE_NAMES:
        if name not in feats:
            return f"invalid_schema:missing:{name}"
        v = feats[name]
        spec = FEATURE_SPEC[name]
        kind = spec["kind"]
        if kind == "numeric":
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return f"invalid_schema:type:{name}"
            lo, hi = _NUMERIC_RANGES.get(name, (None, None))
            if lo is not None and v < lo:
                return f"invalid_schema:range:{name}"
            if hi is not None and v > hi:
                return f"invalid_schema:range:{name}"
        elif kind == "binary":
            if v not in (0, 1):
                return f"invalid_schema:value:{name}"
        elif kind == "categorical":
            if v not in spec.get("choices", []):
                return f"invalid_schema:choice:{name}"

    actual_copd = row.get("actual_copd")
    actual_alt = row.get("actual_alt")
    has_copd_label = actual_copd in _COPD_VALID
    has_alt_label = (
        isinstance(actual_alt, (int, float))
        and not isinstance(actual_alt, bool)
        and _ALT_RANGE[0] <= actual_alt <= _ALT_RANGE[1]
    )
    if not (has_copd_label or has_alt_label):
        return "invalid_schema:no_actual_labels"
    return None


def _check_imputation(row: dict) -> str | None:
    imputed = row.get("imputed_features") or []
    if imputed:
        return f"any_imputation:{','.join(imputed)}"
    return None


def _check_duplicate(row: dict, seen: set[str]) -> str | None:
    h = feature_hash(row["features"])
    if h in seen:
        return f"duplicate:{h[:8]}"
    return None


def validate_and_filter(
    rows: Iterable[dict], seen_hashes: set[str]
) -> tuple[list[dict], list[dict]]:
    """Apply all three gates. `seen_hashes` is mutated so within-batch dups
    on identical feature dicts also get rejected (only the first survives).

    Returns (accepted, rejected). Each output row has `validation_status` set
    to "accepted" or "rejected:<reason>"."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen = set(seen_hashes)
    for r in rows:
        reason = _check_schema(r) or _check_imputation(r) or _check_duplicate(r, seen)
        if reason:
            rejected.append({**r, "validation_status": f"rejected:{reason}"})
            continue
        accepted_row = {**r, "validation_status": "accepted"}
        seen.add(feature_hash(r["features"]))
        accepted.append(accepted_row)
    return accepted, rejected
