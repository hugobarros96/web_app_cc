"""Agent tool: predict patient outcomes (COPD + ALT) with ask-back + imputation.

Flow expected by the agent (see system prompt in agent/system_prompt.py):

1. Agent calls with `ask_back=True` (the default) and whatever features the user gave.
2. If features are missing the tool returns `status="needs_input"` with the missing
   list ordered by combined SHAP importance. The agent surfaces this to the clinician
   and waits for either more features or an explicit "go ahead / I don't know /
   use defaults".
3. The agent re-calls, either with more features (loop step 2) or with
   `ask_back=False` to accept median/mode imputation.
4. On `ask_back=False`, the tool imputes the missing values and runs both models,
   returning predictions, the per-prediction SHAP top-3, the imputation audit,
   and any natural-language to feature assumptions the agent passes in.

v2 additions:
- Each successful prediction gets a `prediction_id` (UUID short hex) threaded
  through the result; the UI uses it to key feedback widgets and the feedback
  log uses it to dedup on re-submission.
- On success, the tool writes `last_patient` and `last_prediction` into
  SessionContext so the session-context block can summarize them on subsequent
  turns.
- The return dict adds `features_used` (the full feature dict after imputation)
  and `imputed_features` (list of names) so the UI can display the imputation
  warning and the feedback log can decide eligibility for training.
"""
from __future__ import annotations

import uuid

from health_assistant.agent.session_state import get_session_ctx, set_session_ctx
from health_assistant.models.feature_schema import FEATURE_SPEC
from health_assistant.models.predict import (
    impute_missing,
    list_missing_features,
    predict_both,
)


def predict_patient_outcomes(
    features: dict,
    ask_back: bool = True,
    assumptions: list[dict] | None = None,
) -> dict:
    """Predict COPD class and ALT value for a patient.

    Args:
        features: dict of any subset of the 15 model features. Keys must match
            FEATURE_SPEC names; values must satisfy the spec's choices/types.
        ask_back: if True and features are missing, return {status:"needs_input",
            missing:[...]} instead of imputing. If False, impute missing values
            with population median (numeric) or mode (categorical) and predict.
        assumptions: optional list of {raw, mapped} dicts the agent can pass to
            record natural-language to feature mappings (e.g.
            "athlete" -> exercise_frequency=High). Passed through to the response
            for clinician review.

    Returns:
        On needs_input:
            {"status":"needs_input", "missing":[{"feature","type","valid"}, ...]}

        On ok:
            {"status":"ok",
             "prediction_id": "<12-char hex>",
             "features_used": {...},                # full feature dict actually used
             "imputed": [{"feature","value","strategy"}, ...],
             "imputed_features": [str, ...],        # convenience: names only
             "copd": {"prediction","class_scores","top_features"},
             "alt":  {"prediction","interval_80","top_features"},
             "assumptions": [{"raw","mapped"}, ...]}
    """
    missing = list_missing_features(features)

    if missing and ask_back:
        return {
            "status": "needs_input",
            "missing": [
                {
                    "feature": f,
                    "type": FEATURE_SPEC[f]["kind"],
                    "valid": FEATURE_SPEC[f].get("choices"),
                }
                for f in missing
            ],
        }

    completed = dict(features)
    imputed: list[dict] = []
    if missing:
        completed, imputed = impute_missing(features)

    preds = predict_both(completed)
    prediction_id = uuid.uuid4().hex[:12]
    imputed_names = [a["feature"] for a in imputed]

    # Update session memory so the next turn's session-context block can summarize
    # this prediction and the UI can attach feedback widgets keyed on prediction_id.
    ctx = get_session_ctx()
    ctx.last_patient = dict(completed)
    ctx.last_prediction = {
        "copd": preds["copd"],
        "alt": preds["alt"],
        "prediction_id": prediction_id,
        "imputed_features": imputed_names,
    }
    set_session_ctx(ctx)

    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "features_used": dict(completed),
        "imputed": imputed,
        "imputed_features": imputed_names,
        "copd": preds["copd"],
        "alt": preds["alt"],
        "assumptions": assumptions or [],
    }
