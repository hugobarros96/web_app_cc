"""Train the COPD multiclass classifier.

Strategy:
- Stratified 80/20 train/test split.
- Optuna over XGBoost hyperparams, 30 trials, 5-fold stratified CV inside the train split.
- Each trial logs to MLflow as a NESTED run under the parent 'copd_xgb' run.
- No isotonic calibration: predict_proba outputs are surfaced as 'relative scores',
  not calibrated probabilities (POC-grade; reflects the EDA finding that COPD has
  no clear feature signal in this synthetic dataset, so calibration would just be
  cosmetic).
- SHAP TreeExplainer for global + per-prediction explanations.
- Saves artifacts to artifacts/models/ and SHAP summary PNG to artifacts/reports/.
"""
from __future__ import annotations

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from health_assistant.config import load_patient_csv, paths, settings
from health_assistant.models.feature_schema import FEATURE_NAMES, TARGET_CSV_COLUMNS
from health_assistant.models.preprocessing import build_preprocessor

SEED = 42
DEFAULT_N_TRIALS = 30
N_SPLITS = 5
TARGET_COL = TARGET_CSV_COLUMNS["chronic_opd"]


def _objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "random_state": SEED,
        "verbosity": 0,
        "n_jobs": 4,  # avoid thrash on machines with many cores
    }

    # Wrap the ENTIRE trial (CV + logging) in the nested MLflow run so
    # the run's duration reflects real training time.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        print(f"[trial {trial.number}] params={params}", flush=True)

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(cv.split(X, y)):
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            s = float(f1_score(y[va], pred, average="macro"))
            scores.append(s)
            mlflow.log_metric(f"fold_{fold_idx}_macro_f1", s)
            print(f"[trial {trial.number}] fold {fold_idx}: macro_f1={s:.4f}", flush=True)

        cv_score = float(np.mean(scores))
        mlflow.log_metric("cv_macro_f1", cv_score)
        mlflow.log_metric("cv_macro_f1_std", float(np.std(scores)))
        mlflow.log_metric("cv_macro_f1_min", float(np.min(scores)))
        mlflow.log_metric("cv_macro_f1_max", float(np.max(scores)))
        print(f"[trial {trial.number}] cv_macro_f1={cv_score:.4f}", flush=True)
    return cv_score


def _save_shap_summary(model: xgb.XGBClassifier, X_sample: np.ndarray, feature_names, path: Path):
    """Return the explainer; also write the summary plot to `path`.
    Returns None if SHAP fails (we keep training resilient to SHAP/XGBoost
    version skew - the model is already saved before this is called)."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        # multiclass: shap_values is a 3D array (n_samples, n_features, n_classes) in newer SHAP
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


def train_copd(n_trials: int = DEFAULT_N_TRIALS) -> dict:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    df = load_patient_csv()
    y_raw = df[TARGET_COL]
    X = df[FEATURE_NAMES]
    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)

    pre = build_preprocessor()  # full 14 features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    with mlflow.start_run(run_name="copd_xgb"):
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("features", ",".join(FEATURE_NAMES))
        mlflow.log_param("target", TARGET_COL)

        # Optuna sweep - each trial is a nested run
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(lambda t: _objective(t, Xt_train, y_train), n_trials=n_trials)

        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_macro_f1", float(study.best_value))

        # Final model on all training data
        final = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            verbosity=0,
            **best,
        )
        final.fit(Xt_train, y_train)

        # Test metrics (raw predict_proba - "relative scores", uncalibrated)
        proba = final.predict_proba(Xt_test)
        pred = np.argmax(proba, axis=1)
        macro_f1 = float(f1_score(y_test, pred, average="macro"))
        ll = float(log_loss(y_test, proba, labels=list(range(4))))
        report = classification_report(
            y_test, pred, target_names=list(label_encoder.classes_), output_dict=True
        )
        cm = confusion_matrix(y_test, pred).tolist()

        mlflow.log_metric("test_macro_f1", macro_f1)
        mlflow.log_metric("test_log_loss", ll)
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    mlflow.log_metric(f"test_{cls}_{k}", float(v))

        # Save model artifacts FIRST (so SHAP failure doesn't lose training)
        joblib.dump(pre, paths.models_dir / "preprocessor.joblib")
        joblib.dump(final, paths.models_dir / "copd_xgb.joblib")
        joblib.dump(label_encoder, paths.models_dir / "copd_label_encoder.joblib")
        mlflow.xgboost.log_model(final, name="copd_xgb")

        # SHAP global summary plot (best-effort; resilient to version skew)
        sample = Xt_test[: min(500, len(Xt_test))]
        feature_names_out = list(pre.get_feature_names_out())
        summary_path = paths.reports_dir / "shap_copd_summary.png"
        explainer = _save_shap_summary(final, sample, feature_names_out, summary_path)
        if explainer is not None:
            joblib.dump(explainer, paths.models_dir / "shap_explainer_copd.pkl")
            if summary_path.exists():
                mlflow.log_artifact(str(summary_path))

        result = {
            "macro_f1": macro_f1,
            "log_loss": ll,
            "best_params": best,
            "confusion_matrix": cm,
            "classes": list(label_encoder.classes_),
        }
        metrics_path = paths.models_dir / "copd_metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(metrics_path))
        return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                    help=f"Number of Optuna trials (default: {DEFAULT_N_TRIALS})")
    args = ap.parse_args()
    out = train_copd(n_trials=args.n_trials)
    print(json.dumps(out, indent=2))
