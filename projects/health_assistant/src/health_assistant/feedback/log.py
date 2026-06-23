"""Append-only feedback log + eligibility counter + retrain trigger.

Pipeline:
1. Streamlit UI (Task 6.4) calls log_feedback(...) when the user clicks 👍 or
   submits a correction.
2. Eligibility (eligible_for_training=True) is computed at write time:
   requires zero imputed features AND at least one actual label.
3. Counter (`eligibility_counts`) reads the file and reports how many eligible
   rows have not yet been consumed by a retrain.
4. When the eligible count crosses RETRAIN_THRESHOLD, maybe_trigger_retrain()
   fires retrain_with_feedback() in a background daemon thread.

The file is rewritten on every write (read-all, dedup-by-prediction_id,
write-all) so re-submitting feedback for the same prediction_id overwrites
rather than duplicates. Cheap for POC scale (hundreds of rows at most);
production would use append-only with a separate compaction job.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from health_assistant.config import paths

FEEDBACK_DIR = paths.artifacts_dir / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
RETRAIN_THRESHOLD = 5

_TRIGGER_LOCK = threading.Lock()


def _ensure_dir() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def log_feedback(
    *,
    prediction_id: str,
    session_id: str,
    features: dict,
    imputed_features: list[str],
    predicted_copd: str,
    predicted_copd_scores: dict,
    predicted_alt: float,
    predicted_alt_interval_80: list[float],
    actual_copd: str | None,
    actual_alt: float | None,
    kind: str,  # "thumbs_up" | "correction"
) -> dict:
    """Append (or overwrite by prediction_id) a feedback row.

    Returns the row written. Eligibility is set here: a row is
    eligible_for_training only when no features were imputed AND at least one
    of actual_copd / actual_alt is non-null.
    """
    _ensure_dir()
    eligible = (
        len(imputed_features) == 0
        and (actual_copd is not None or actual_alt is not None)
    )
    row = {
        "prediction_id": prediction_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": features,
        "imputed_features": imputed_features,
        "predicted_copd": predicted_copd,
        "predicted_copd_scores": predicted_copd_scores,
        "predicted_alt": predicted_alt,
        "predicted_alt_interval_80": predicted_alt_interval_80,
        "actual_copd": actual_copd,
        "actual_alt": actual_alt,
        "kind": kind,
        "eligible_for_training": eligible,
        "consumed_by_retrain": None,
        "validation_status": None,
    }
    # Dedup-by-prediction_id: drop any prior row with the same id, append fresh.
    rows = _read_all()
    rows = [r for r in rows if r.get("prediction_id") != prediction_id]
    rows.append(row)
    _write_all(rows)
    return row


def _read_all() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    out: list[dict] = []
    for line in FEEDBACK_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # Skip malformed lines; never crash the writer because of a corrupt row.
            continue
    return out


def _write_all(rows: list[dict]) -> None:
    """Atomic-ish rewrite: write to a tmp file then replace, so a crash mid-
    write leaves the previous file intact."""
    _ensure_dir()
    tmp = FEEDBACK_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    tmp.replace(FEEDBACK_FILE)


def eligibility_counts() -> dict:
    """Current state of the AL panel counters. Cheap to call on every rerun."""
    rows = _read_all()
    eligible_pending = [
        r for r in rows
        if r.get("eligible_for_training") and r.get("consumed_by_retrain") is None
    ]
    return {
        "total": len(rows),
        "eligible_pending": len(eligible_pending),
        "threshold": RETRAIN_THRESHOLD,
    }


def maybe_trigger_retrain() -> bool:
    """Fire retrain_with_feedback in a background thread when threshold is met.

    Returns True iff a retrain was actually started this call; False otherwise.
    Re-entrant safe: the retrain implementation acquires a file lock and refuses
    to run more than one at a time, so a second trigger arriving before the
    first finishes will be no-op'd inside retrain itself.
    """
    counts = eligibility_counts()
    if counts["eligible_pending"] < RETRAIN_THRESHOLD:
        return False

    # Import inside the function so log.py doesn't pull in xgboost / sklearn
    # at import time (the retrain module is heavy).
    from health_assistant.models.retrain import retrain_with_feedback

    def _thread():
        try:
            retrain_with_feedback()
        except Exception as e:  # pragma: no cover - background safety
            print(f"[feedback] retrain crashed: {e}")

    with _TRIGGER_LOCK:
        t = threading.Thread(target=_thread, daemon=True)
        t.start()
    return True
