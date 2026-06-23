"""Feedback-driven retrain: validate, fold, train, evaluate, promote.

Triggered by feedback.log.maybe_trigger_retrain() once 5 eligible feedback
rows accumulate. Reuses the best hyperparams the initial Optuna study found
(loaded from copd_metrics.json and alt_metrics.json) - no Optuna re-tuning,
single training run per model, finishes in ~30 sec on CPU.

Atomic promotion gate:
  - COPD: new_macro_f1_holdout >= production - EPSILON_COPD (default 0.005)
  - ALT:  new_mae_holdout       <= production + EPSILON_ALT_MAE (default 0.05)

On promotion, the previous model is archived to artifacts/models/archive/<ts>/
before the candidate is renamed into place via os.replace (atomic on POSIX),
production_metrics.json is updated, and predict.reset_model_caches() is called
so live sessions immediately pick up the new model.

Concurrency safety: a sentinel lock file is acquired at the start and released
in a try/finally. If a second retrain call arrives while one is running, it
short-circuits with status="skipped_in_progress".

Use `dry_run=True` (test path) to run validation + training + evaluation but
skip every disk mutation that would touch production (archive, swap,
production_metrics, consumed_by_retrain stamps, cache invalidation).
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from health_assistant.config import load_patient_csv, paths
from health_assistant.feedback import log as fb_log
from health_assistant.feedback.validation import feature_hash, validate_and_filter
from health_assistant.models.feature_schema import FEATURE_NAMES

SEEN_HASHES_FILE = paths.artifacts_dir / "feedback" / "seen_hashes.jsonl"
STATUS_FILE = paths.artifacts_dir / "feedback" / "retrain_status.json"
LOCK_FILE = paths.artifacts_dir / "feedback" / ".retrain.lock"
PRODUCTION_METRICS = paths.models_dir / "production_metrics.json"
EPSILON_COPD = 0.005
EPSILON_ALT_MAE = 0.05
HOLDOUT_SEED = 42


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------
def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        return False
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(time.time()))
    return True


def _release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Status file (current + append-history)
# ---------------------------------------------------------------------------
def _write_status(payload: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    history: list = []
    if STATUS_FILE.exists():
        try:
            history = json.loads(STATUS_FILE.read_text()).get("history", [])
        except Exception:
            history = []
    history.append(payload)
    STATUS_FILE.write_text(json.dumps({"current": payload, "history": history}, indent=2))


# ---------------------------------------------------------------------------
# Seen-hashes registry (training-data hashes + accepted feedback hashes)
# ---------------------------------------------------------------------------
def _read_seen_hashes() -> set[str]:
    if not SEEN_HASHES_FILE.exists():
        return set()
    return {ln.strip() for ln in SEEN_HASHES_FILE.read_text().splitlines() if ln.strip()}


def _append_seen_hashes(new: set[str]) -> None:
    if not new:
        return
    SEEN_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES_FILE.open("a") as f:
        for h in new:
            f.write(h + "\n")


def _seed_seen_from_training_data(df: pd.DataFrame) -> None:
    """First-time seed: hash every existing training row so feedback that
    duplicates a CSV row gets rejected."""
    if SEEN_HASHES_FILE.exists():
        return
    SEEN_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES_FILE.open("w") as f:
        for _, row in df[FEATURE_NAMES].iterrows():
            f.write(feature_hash(row.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Production metrics (current + retrain history)
# ---------------------------------------------------------------------------
def _load_production_metrics() -> dict:
    """Return the current production-model metrics. On first run, seeds from
    the initial-training metrics files written by train_copd / train_alt."""
    if PRODUCTION_METRICS.exists():
        return json.loads(PRODUCTION_METRICS.read_text())
    copd_init = json.loads((paths.models_dir / "copd_metrics.json").read_text())
    alt_init = json.loads((paths.models_dir / "alt_metrics.json").read_text())
    return {
        "copd": {
            "macro_f1_holdout": copd_init["macro_f1"],
            "best_params": copd_init["best_params"],
        },
        "alt": {
            "mae_holdout": alt_init["mae"],
            "r2_holdout": alt_init["r2"],
            "best_params": alt_init["best_params"],
        },
        "epsilon_copd_macro_f1": EPSILON_COPD,
        "epsilon_alt_mae": EPSILON_ALT_MAE,
        "feedback_retrains": [],
    }


def _save_production_metrics(payload: dict) -> None:
    PRODUCTION_METRICS.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Training (reuse best hyperparams; no Optuna)
# ---------------------------------------------------------------------------
def _train_copd(X_train, y_train, params: dict):
    from xgboost import XGBClassifier

    # Reuse the existing fitted preprocessor + label encoder so the candidate
    # has the same feature ordering as the production model.
    pre = joblib.load(paths.models_dir / "preprocessor.joblib")
    le = joblib.load(paths.models_dir / "copd_label_encoder.joblib")

    Xt = pre.transform(X_train[FEATURE_NAMES])
    yt = le.transform(y_train.astype(str))
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_jobs=4,
        random_state=42,
        **params,
    )
    model.fit(Xt, yt)
    return model, pre, le


def _train_alt(X_train, y_train, params: dict):
    from xgboost import XGBRegressor

    pre = joblib.load(paths.models_dir / "alt_preprocessor.joblib")
    Xt = pre.transform(X_train[FEATURE_NAMES])
    model = XGBRegressor(
        objective="reg:squarederror", n_jobs=4, random_state=42, **params
    )
    model.fit(Xt, y_train)
    return model, pre


def _eval_copd(model, pre, le, X_holdout, y_holdout) -> float:
    Xt = pre.transform(X_holdout[FEATURE_NAMES])
    pred_idx = model.predict(Xt)
    pred_labels = le.inverse_transform(pred_idx)
    return float(
        f1_score(y_holdout.astype(str), pred_labels.astype(str), average="macro")
    )


def _eval_alt(model, pre, X_holdout, y_holdout) -> tuple[float, float]:
    Xt = pre.transform(X_holdout[FEATURE_NAMES])
    pred = model.predict(Xt)
    return (
        float(mean_absolute_error(y_holdout, pred)),
        float(r2_score(y_holdout, pred)),
    )


def _archive_current_models() -> None:
    """Snapshot the current production model files into archive/<ts>/ so a
    promoted model can be manually rolled back if it turns out to be bad."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive = paths.models_dir / "archive" / ts
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("copd_xgb.joblib", "alt_xgb.joblib"):
        src = paths.models_dir / name
        if src.exists():
            shutil.copy2(src, archive / name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def retrain_with_feedback(dry_run: bool = False) -> dict:
    """Validate + fold + train + evaluate + (optionally) promote.

    See module docstring for the full pipeline. `dry_run=True` runs every step
    that doesn't mutate production state - used by tests to exercise the
    pipeline without overwriting real model files.

    Returns a dict whose `status` field is one of:
        skipped_in_progress | no_pending | insufficient_valid_rows
        | promoted | rejected
        | dry_run_promoted | dry_run_rejected
        | failed
    """
    if not _acquire_lock():
        payload = {
            "status": "skipped_in_progress",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not dry_run:
            _write_status(payload)
        return payload

    try:
        rows = fb_log._read_all()
        pending = [
            r for r in rows
            if r.get("eligible_for_training") and r.get("consumed_by_retrain") is None
        ]
        if not pending:
            payload = {
                "status": "no_pending",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if not dry_run:
                _write_status(payload)
            return payload

        df = load_patient_csv()
        _seed_seen_from_training_data(df)
        seen = _read_seen_hashes()

        accepted, rejected = validate_and_filter(pending, seen)
        rej_breakdown: dict[str, int] = {}
        for r in rejected:
            reason = r["validation_status"].split(":", 2)[1]
            rej_breakdown[reason] = rej_breakdown.get(reason, 0) + 1

        if len(accepted) < 3:
            payload = {
                "status": "insufficient_valid_rows",
                "rows_submitted": len(pending),
                "rows_accepted": len(accepted),
                "rows_rejected": len(rejected),
                "rejection_breakdown": rej_breakdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if not dry_run:
                _write_status(payload)
            return payload

        # Deterministic split so candidate is evaluated on the same holdout
        # used as the production baseline.
        train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=HOLDOUT_SEED)

        # Fold accepted feedback into training data. Each row contributes a
        # synthetic patient. ALT label falls back to the model's prediction when
        # the user only corrected the COPD label (so the COPD model still sees
        # the row without polluting the ALT target with a wrong label).
        new_rows: list[dict] = []
        for r in accepted:
            base = {**r["features"]}
            if r.get("actual_copd"):
                base["chronic_obstructive_pulmonary_disease"] = r["actual_copd"]
            else:
                # No COPD label - reuse the predicted class so the row at least
                # has a value; this row will only get folded into ALT training.
                base["chronic_obstructive_pulmonary_disease"] = r["predicted_copd"]
            base["alanine_aminotransferase"] = (
                r["actual_alt"] if r.get("actual_alt") is not None else r["predicted_alt"]
            )
            new_rows.append(base)
        feedback_df = pd.DataFrame(new_rows)

        # Only feed COPD training with rows that actually have a user COPD label
        copd_feedback = feedback_df[
            feedback_df.index.isin(
                [i for i, r in enumerate(accepted) if r.get("actual_copd")]
            )
        ]
        alt_feedback = feedback_df[
            feedback_df.index.isin(
                [i for i, r in enumerate(accepted) if r.get("actual_alt") is not None]
            )
        ]
        combined_train_copd = pd.concat([train_df, copd_feedback], ignore_index=True)
        combined_train_alt = pd.concat([train_df, alt_feedback], ignore_index=True)

        prod = _load_production_metrics()

        copd_model, copd_pre, copd_le = _train_copd(
            combined_train_copd,
            combined_train_copd["chronic_obstructive_pulmonary_disease"],
            prod["copd"]["best_params"],
        )
        alt_model, alt_pre = _train_alt(
            combined_train_alt,
            combined_train_alt["alanine_aminotransferase"],
            prod["alt"]["best_params"],
        )

        new_macro_f1 = _eval_copd(
            copd_model, copd_pre, copd_le,
            holdout_df, holdout_df["chronic_obstructive_pulmonary_disease"],
        )
        new_alt_mae, new_alt_r2 = _eval_alt(
            alt_model, alt_pre,
            holdout_df, holdout_df["alanine_aminotransferase"],
        )

        copd_promoted = new_macro_f1 >= (prod["copd"]["macro_f1_holdout"] - EPSILON_COPD)
        alt_promoted = new_alt_mae <= (prod["alt"]["mae_holdout"] + EPSILON_ALT_MAE)

        retrain_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rows_submitted": len(pending),
            "rows_accepted": len(accepted),
            "rows_rejected": len(rejected),
            "rejection_breakdown": rej_breakdown,
            "copd_promoted": bool(copd_promoted),
            "alt_promoted": bool(alt_promoted),
            "delta_macro_f1": float(new_macro_f1 - prod["copd"]["macro_f1_holdout"]),
            "delta_alt_mae": float(prod["alt"]["mae_holdout"] - new_alt_mae),
            "new_macro_f1": float(new_macro_f1),
            "new_alt_mae": float(new_alt_mae),
            "new_alt_r2": float(new_alt_r2),
        }

        if dry_run:
            status = "dry_run_promoted" if (copd_promoted or alt_promoted) else "dry_run_rejected"
            payload = {"status": status, **retrain_record}
            return payload

        # ----- mutating section (real runs only) ------------------------------
        if copd_promoted or alt_promoted:
            _archive_current_models()

        if copd_promoted:
            candidate_path = paths.models_dir / "copd_xgb.candidate.joblib"
            joblib.dump(copd_model, candidate_path)
            candidate_path.replace(paths.models_dir / "copd_xgb.joblib")
            prod["copd"]["macro_f1_holdout"] = float(new_macro_f1)

        if alt_promoted:
            candidate_path = paths.models_dir / "alt_xgb.candidate.joblib"
            joblib.dump(alt_model, candidate_path)
            candidate_path.replace(paths.models_dir / "alt_xgb.joblib")
            prod["alt"]["mae_holdout"] = float(new_alt_mae)
            prod["alt"]["r2_holdout"] = float(new_alt_r2)

        prod["feedback_retrains"].append(retrain_record)
        _save_production_metrics(prod)

        # Stamp consumed_by_retrain on the rows that were processed (accepted
        # AND rejected; rejected ones don't re-trigger validation next round).
        accepted_ids = {r["prediction_id"] for r in accepted}
        rejected_id_to_status = {r["prediction_id"]: r["validation_status"] for r in rejected}
        retrain_id = retrain_record["timestamp"]

        all_rows = fb_log._read_all()
        new_seen_hashes: set[str] = set()
        for r in all_rows:
            pid = r.get("prediction_id")
            if pid in accepted_ids:
                r["consumed_by_retrain"] = retrain_id
                r["validation_status"] = "accepted"
                new_seen_hashes.add(feature_hash(r["features"]))
            elif pid in rejected_id_to_status:
                r["consumed_by_retrain"] = retrain_id
                r["validation_status"] = rejected_id_to_status[pid]
        fb_log._write_all(all_rows)
        _append_seen_hashes(new_seen_hashes)

        # Invalidate predict.py caches so live sessions pick up the new model.
        from health_assistant.models.predict import reset_model_caches

        reset_model_caches()

        status = "promoted" if (copd_promoted or alt_promoted) else "rejected"
        payload = {"status": status, **retrain_record}
        _write_status(payload)
        return payload
    finally:
        _release_lock()
