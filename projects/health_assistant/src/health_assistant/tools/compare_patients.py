"""compare_patients - look up 2-5 patients by patient_id, return side-by-side
table + per-patient predictions + a bar-chart PNG (rendered inline)."""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for background tools
import matplotlib.pyplot as plt

from health_assistant.config import load_patient_csv
from health_assistant.models.feature_schema import FEATURE_NAMES, TARGET_CSV_COLUMNS
from health_assistant.models.predict import impute_missing, predict_both

_COPD_COL = TARGET_CSV_COLUMNS["chronic_opd"]
_ALT_COL = TARGET_CSV_COLUMNS["alt"]
from health_assistant.tools.python_analytics import _figures_lock, _figures_this_turn

# Numeric features shown in the bar chart (categoricals don't compare well as bars).
_CHART_FEATURES = ["age", "bmi", "medication_count", "days_hospitalized", "last_lab_glucose"]


def compare_patients(patient_ids: list[str], include_predictions: bool = True) -> dict:
    """Compare 2-5 real patients side-by-side.

    Returns:
        On error: {"error": str}
        On success: {
            "table": [{"patient_id", ...all features in FEATURE_NAMES}, ...],
            "predictions": [{"patient_id", "copd_recorded", "alt_recorded",
                             "copd_class", "copd_scores", "alt_value",
                             "alt_interval_80"}, ...] | None,
            "chart_b64": "<png base64>"
        }
    """
    if not isinstance(patient_ids, list) or not all(isinstance(p, str) for p in patient_ids):
        return {"error": "patient_ids must be a list of strings (e.g. ['P00042', 'P00115'])."}
    if not (2 <= len(patient_ids) <= 5):
        return {"error": f"patient_ids must be between 2 and 5 items; got {len(patient_ids)}."}

    df = load_patient_csv()
    missing = [pid for pid in patient_ids if pid not in df["patient_id"].values]
    if missing:
        return {"error": f"Unknown patient_id(s): {missing}. Use python_analytics to list valid IDs."}

    # Preserve user-requested order and dedup.
    rows = df[df["patient_id"].isin(patient_ids)].drop_duplicates("patient_id").set_index("patient_id")
    rows = rows.reindex(patient_ids)

    table = [
        {"patient_id": pid, **{f: _to_jsonable(rows.loc[pid, f]) for f in FEATURE_NAMES if f in rows.columns}}
        for pid in patient_ids
    ]

    predictions = None
    if include_predictions:
        predictions = []
        for pid in patient_ids:
            features = {f: rows.loc[pid, f] for f in FEATURE_NAMES if f in rows.columns}
            # Some CSV rows have NaN in exercise_frequency / education_level
            # (genuine missing data). The model encoder can't handle NaN -
            # impute via population mode/median before predicting.
            completed, _imputed = impute_missing(features)
            preds = predict_both(completed)
            predictions.append({
                "patient_id": pid,
                # Recorded ground truth from the dataset, shown alongside the
                # model output so real patients aren't misrepresented by a
                # (near-random) COPD prediction. None if the column is absent.
                "copd_recorded": _to_jsonable(rows.loc[pid, _COPD_COL]) if _COPD_COL in rows.columns else None,
                "alt_recorded": _to_jsonable(rows.loc[pid, _ALT_COL]) if _ALT_COL in rows.columns else None,
                "copd_class": preds["copd"]["prediction"],
                "copd_scores": preds["copd"]["class_scores"],
                "alt_value": float(preds["alt"]["prediction"]),
                "alt_interval_80": [float(x) for x in preds["alt"]["interval_80"]],
            })

    chart_b64 = _build_chart(rows, patient_ids)
    # Push to the per-turn figure registry so the Streamlit UI renders it
    # alongside the agent's reply (same path as python_analytics charts).
    # We deliberately DO NOT include the base64 in the dict returned to the LLM:
    # the model would dutifully try to inline tens of thousands of tokens of PNG
    # data as markdown and hit max_tokens. The chart_rendered + chart_features
    # fields tell the agent what to expect in the rendered figure.
    with _figures_lock:
        _figures_this_turn.append(chart_b64)
    chart_features = [k for k in _CHART_FEATURES if k in rows.columns]
    return {
        "table": table,
        "predictions": predictions,
        "chart_rendered": True,
        "chart_features": chart_features,
    }


def _to_jsonable(v):
    """Cast numpy scalars to plain Python so the JSON the agent sees is clean."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def _build_chart(rows, patient_ids: list[str]) -> str:
    """One subplot per numeric feature, one bar per patient. Returns base64 PNG."""
    keys = [k for k in _CHART_FEATURES if k in rows.columns]
    if not keys:
        # Fallback: empty 1x1 placeholder
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.axis("off")
    else:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.5))
        if len(keys) == 1:
            axes = [axes]
        for ax, feat in zip(axes, keys):
            values = [float(rows.loc[pid, feat]) for pid in patient_ids]
            ax.bar(patient_ids, values)
            ax.set_title(feat, fontsize=10)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
