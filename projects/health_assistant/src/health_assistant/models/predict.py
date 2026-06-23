"""Inference for both COPD and ALT models.

Surface (consumed by the `predict_patient_outcomes` tool):
- `predict_both(features)` - runs both models, returns combined result.
- `list_missing_features(features)` - returns missing features ordered by combined SHAP importance.
- `impute_missing(features)` - fills missing features with median/mode and reports what it imputed.
- `feature_importance_order()` - the ordering used for ask-back.

The ALT model uses all 15 features (including BMI); BMI carries nearly all of
ALT's signal in this synthetic dataset (corr ≈ 0.9998), so the model reaches
R² ≈ 0.999.

`class_scores` for COPD are the raw XGBoost softmax outputs - they are *relative
likelihoods*, NOT calibrated probabilities (see spec D-row for context).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from health_assistant.config import load_patient_csv, paths
from health_assistant.models.feature_schema import (
    FEATURE_NAMES,
    FEATURE_SPEC,
)


@lru_cache(maxsize=1)
def _artifacts() -> dict[str, Any]:
    return {
        # COPD - uses all 15 features
        "copd_pre": joblib.load(paths.models_dir / "preprocessor.joblib"),
        "copd_model": joblib.load(paths.models_dir / "copd_xgb.joblib"),
        "copd_le": joblib.load(paths.models_dir / "copd_label_encoder.joblib"),
        "copd_shap": _try_load(paths.models_dir / "shap_explainer_copd.pkl"),
        # ALT - all 15 features (including BMI).
        "alt_pre": joblib.load(paths.models_dir / "alt_preprocessor.joblib"),
        "alt_mean": joblib.load(paths.models_dir / "alt_xgb.joblib"),
        "alt_q10": joblib.load(paths.models_dir / "alt_xgb_q10.joblib"),
        "alt_q90": joblib.load(paths.models_dir / "alt_xgb_q90.joblib"),
        "alt_shap": _try_load(paths.models_dir / "shap_explainer_alt.pkl"),
        # population data - for imputation defaults
        "df": load_patient_csv(),
    }


def _try_load(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def feature_importance_order() -> tuple[str, ...]:
    """Return FEATURE_NAMES ordered by combined mean(|SHAP|) across both models.

    Cached to a JSON file under artifacts/models/ so subsequent calls are instant
    and so the ordering is reproducible across processes (the agent's ask-back
    sequence is then identical between Streamlit reloads)."""
    cache = paths.models_dir / "feature_importance_order.json"
    if cache.exists():
        return tuple(json.loads(cache.read_text()))

    arts = _artifacts()
    df = arts["df"].head(500)

    importance = {f: 0.0 for f in FEATURE_NAMES}

    # --- COPD contribution (covers all 15 features) ---
    copd_shap = arts["copd_shap"]
    if copd_shap is not None:
        Xt = arts["copd_pre"].transform(df[FEATURE_NAMES])
        sv = copd_shap.shap_values(Xt)  # shape (n_samples, n_features_encoded, n_classes)
        sv_mean = np.abs(np.asarray(sv)).mean(axis=(0, -1))  # collapse samples + classes
        encoded = list(arts["copd_pre"].get_feature_names_out())
        for i, enc in enumerate(encoded):
            for orig in FEATURE_NAMES:
                if enc == orig or enc.startswith(orig + "_"):
                    importance[orig] += float(sv_mean[i])
                    break

    # --- ALT contribution (all 15 features) ---
    alt_shap = arts["alt_shap"]
    if alt_shap is not None:
        Xt = arts["alt_pre"].transform(df[FEATURE_NAMES])
        sv = alt_shap.shap_values(Xt)  # shape (n_samples, n_features_encoded)
        sv_mean = np.abs(np.asarray(sv)).mean(axis=0)
        encoded = list(arts["alt_pre"].get_feature_names_out())
        for i, enc in enumerate(encoded):
            for orig in FEATURE_NAMES:
                if enc == orig or enc.startswith(orig + "_"):
                    importance[orig] += float(sv_mean[i])
                    break

    ordered = sorted(FEATURE_NAMES, key=lambda f: -importance[f])
    cache.write_text(json.dumps(ordered))
    return tuple(ordered)


def _is_missing(v) -> bool:
    """A feature value counts as missing if absent, NaN, or empty string."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v == "":
        return True
    return False


def list_missing_features(features: dict) -> list[str]:
    """Missing features in importance order (highest importance first).

    A feature counts as missing if its key is absent OR its value is None /
    NaN / empty string (the CSV's missing-data representations for
    exercise_frequency and education_level)."""
    return [
        f for f in feature_importance_order()
        if f not in features or _is_missing(features[f])
    ]


def impute_missing(features: dict) -> tuple[dict, list[dict]]:
    """Fill missing features with population median (numeric) or mode
    (categorical/binary). Returns (completed_features, audit_list).

    "Missing" means absent key OR NaN/None/empty-string value."""
    df = _artifacts()["df"]
    completed = dict(features)
    audit: list[dict] = []
    for name in FEATURE_NAMES:
        if name in completed and not _is_missing(completed[name]):
            continue
        spec = FEATURE_SPEC[name]
        if spec["kind"] in ("numeric",):
            val = float(df[name].median())
            strategy = "median"
        elif spec["kind"] == "binary":
            val = int(df[name].mode().iloc[0])
            strategy = "mode"
        else:  # categorical
            val = str(df[name].mode().iloc[0])
            strategy = "mode"
        completed[name] = val
        audit.append({"feature": name, "value": val, "strategy": strategy})
    return completed, audit


def _collapse_shap(
    shap_row: np.ndarray,
    encoded_names: list[str],
    original_names: list[str],
) -> list[dict]:
    """Collapse one-hot-expanded SHAP values back to original feature names.
    Uses max(|shap|) across the encoded columns that derive from the same original."""
    scores: dict[str, float] = {f: 0.0 for f in original_names}
    for i, enc in enumerate(encoded_names):
        for orig in original_names:
            if enc == orig or enc.startswith(orig + "_"):
                scores[orig] = max(scores[orig], abs(float(shap_row[i])))
                break
    return sorted(
        ({"feature": f, "shap": s} for f, s in scores.items()),
        key=lambda d: -d["shap"],
    )


def reset_model_caches() -> None:
    """Invalidate the lru_cache on artifact loading so the next predict_both()
    call reloads the model files from disk. Called by retrain.py after an
    atomic swap so live sessions immediately see the new model."""
    _artifacts.cache_clear()
    feature_importance_order.cache_clear()


def predict_both(features: dict) -> dict:
    """Run both models. Assumes all 15 model features are present.
    Use list_missing_features() / impute_missing() upstream if not."""
    a = _artifacts()

    # ---- COPD (multiclass, all 15 features) ----
    row_copd = pd.DataFrame([{k: features[k] for k in FEATURE_NAMES}])
    Xt_copd = a["copd_pre"].transform(row_copd)
    proba = a["copd_model"].predict_proba(Xt_copd)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(a["copd_le"].inverse_transform([pred_idx])[0])
    class_scores = {str(c): float(p) for c, p in zip(a["copd_le"].classes_, proba)}

    top_copd: list[dict] = []
    if a["copd_shap"] is not None:
        sv = np.asarray(a["copd_shap"].shap_values(Xt_copd))  # (1, n_enc, n_classes)
        shap_for_pred = sv[0, :, pred_idx]  # for the predicted class
        top_copd = _collapse_shap(shap_for_pred, list(a["copd_pre"].get_feature_names_out()), FEATURE_NAMES)[:3]

    # ---- ALT (regression, all 15 features including BMI) ----
    row_alt = pd.DataFrame([{k: features[k] for k in FEATURE_NAMES}])
    Xt_alt = a["alt_pre"].transform(row_alt)
    alt_mean_val = float(a["alt_mean"].predict(Xt_alt)[0])
    q10 = float(a["alt_q10"].predict(Xt_alt)[0])
    q90 = float(a["alt_q90"].predict(Xt_alt)[0])

    top_alt: list[dict] = []
    if a["alt_shap"] is not None:
        sv = np.asarray(a["alt_shap"].shap_values(Xt_alt))  # (1, n_enc)
        top_alt = _collapse_shap(sv[0], list(a["alt_pre"].get_feature_names_out()), FEATURE_NAMES)[:3]

    return {
        "copd": {
            "prediction": pred_label,
            "class_scores": class_scores,
            "top_features": top_copd,
        },
        "alt": {
            "prediction": alt_mean_val,
            "interval_80": [min(q10, q90), max(q10, q90)],
            "top_features": top_alt,
        },
    }
