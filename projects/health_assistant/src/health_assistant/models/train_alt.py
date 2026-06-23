"""Train the ALT regressor on all 15 features (including BMI).

The model ships a mean head plus two quantile heads (q=0.1, q=0.9) for an 80%
prediction interval. The EDA showed corr(BMI, ALT) = 0.9998, so BMI carries
essentially all of ALT's signal in this synthetic dataset - the model leans on
it heavily and reaches R² ≈ 0.999. (With real data we'd verify whether the
BMI↔ALT relationship is genuine physiology before trusting it that far.)

Training runs through Optuna (30 trials by default, CLI-configurable) with
nested MLflow runs per trial for full UI visibility. SHAP is wrapped in
try/except so a version-skew failure cannot wipe out a successful training run.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from health_assistant.config import load_patient_csv, paths, settings
from health_assistant.models.feature_schema import (
    FEATURE_NAMES,
    TARGET_CSV_COLUMNS,
)
from health_assistant.models.preprocessing import build_preprocessor

SEED = 42
DEFAULT_N_TRIALS = 30
N_SPLITS = 5
TARGET_COL = TARGET_CSV_COLUMNS["alt"]


def _objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Return mean CV RMSE (lower is better). Nested MLflow run wraps the whole trial."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "random_state": SEED,
        "verbosity": 0,
        "n_jobs": 4,
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        rmses: list[float] = []
        for fold_idx, (tr, va) in enumerate(cv.split(X)):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            r = float(np.sqrt(mean_squared_error(y[va], pred)))
            rmses.append(r)
            mlflow.log_metric(f"fold_{fold_idx}_rmse", r)
        cv_rmse = float(np.mean(rmses))
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_rmse_std", float(np.std(rmses)))
        mlflow.log_metric("cv_rmse_min", float(np.min(rmses)))
        mlflow.log_metric("cv_rmse_max", float(np.max(rmses)))
        print(f"[trial {trial.number}] cv_rmse={cv_rmse:.4f}", flush=True)
    return cv_rmse


def _save_shap_summary_regression(model: xgb.XGBRegressor, X_sample: np.ndarray, feature_names, path: Path):
    """Best-effort SHAP for the regressor. Returns the explainer or None on failure."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(
            shap_values, X_sample, feature_names=list(feature_names),
            show=False, plot_size=(10, 6),
        )
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
        return explainer
    except Exception as e:
        print(f"[WARN] SHAP step failed (model already saved): {type(e).__name__}: {e}")
        return None


def _train_alt_model(df, n_trials: int) -> dict:
    """Train the ALT regressor (mean + quantile heads) on all 15 features.
    Returns a metrics dict and writes the canonical artifacts."""
    print(f"\n{'='*60}\n[ALT] starting - features={len(FEATURE_NAMES)} n_trials={n_trials}\n{'='*60}", flush=True)

    y = df[TARGET_COL].values
    X = df[FEATURE_NAMES]

    pre = build_preprocessor(feature_subset=FEATURE_NAMES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    with mlflow.start_run(run_name="alt_xgb"):
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("features", ",".join(FEATURE_NAMES))
        mlflow.log_param("target", TARGET_COL)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(lambda t: _objective(t, Xt_train, y_train), n_trials=n_trials)

        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_rmse", float(study.best_value))

        base_kwargs = dict(
            tree_method="hist",
            random_state=SEED,
            verbosity=0,
            n_jobs=4,
            **best,
        )

        # Mean head
        mean_model = xgb.XGBRegressor(objective="reg:squarederror", **base_kwargs)
        mean_model.fit(Xt_train, y_train)
        pred_mean = mean_model.predict(Xt_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred_mean)))
        mae = float(mean_absolute_error(y_test, pred_mean))
        r2 = float(r2_score(y_test, pred_mean))
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # Save artifacts FIRST so SHAP failure cannot lose training.
        joblib.dump(pre, paths.models_dir / "alt_preprocessor.joblib")
        joblib.dump(mean_model, paths.models_dir / "alt_xgb.joblib")
        mlflow.xgboost.log_model(mean_model, name="alt_xgb")

        # Quantile heads for the 80% prediction interval.
        q_models = {}
        for q in (0.1, 0.9):
            qm = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                **base_kwargs,
            )
            qm.fit(Xt_train, y_train)
            q_models[q] = qm
            joblib.dump(qm, paths.models_dir / f"alt_xgb_q{int(q*100):02d}.joblib")

        # Empirical coverage of the 80% interval
        q10_pred = q_models[0.1].predict(Xt_test)
        q90_pred = q_models[0.9].predict(Xt_test)
        inside = ((y_test >= np.minimum(q10_pred, q90_pred)) &
                  (y_test <= np.maximum(q10_pred, q90_pred))).mean()
        mlflow.log_metric("interval_80_coverage", float(inside))

        # SHAP
        sample = Xt_test[: min(500, len(Xt_test))]
        feature_names_out = list(pre.get_feature_names_out())
        summary_path = paths.reports_dir / "shap_alt_summary.png"
        explainer = _save_shap_summary_regression(mean_model, sample, feature_names_out, summary_path)
        if explainer is not None:
            joblib.dump(explainer, paths.models_dir / "shap_explainer_alt.pkl")
            if summary_path.exists():
                mlflow.log_artifact(str(summary_path))

        result = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_params": best,
            "features": list(FEATURE_NAMES),
        }
        print(f"[ALT] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}", flush=True)
        return result


def train_alt(n_trials: int = DEFAULT_N_TRIALS) -> dict:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    df = load_patient_csv()

    result = _train_alt_model(df, n_trials)
    (paths.models_dir / "alt_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                    help=f"Optuna trials (default: {DEFAULT_N_TRIALS})")
    args = ap.parse_args()
    out = train_alt(n_trials=args.n_trials)
    print(json.dumps(out, indent=2))
