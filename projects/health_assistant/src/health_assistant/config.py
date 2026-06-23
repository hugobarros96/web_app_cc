from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Auto-load env vars from .env at import time. Existing environment variables
# are NOT overridden - explicit env wins. `.env.example` is only a template to
# copy to `.env`; it is never read.
_dotenv = REPO_ROOT / ".env"
if _dotenv.exists():
    load_dotenv(_dotenv, override=False)


def load_patient_csv() -> "pd.DataFrame":
    """Read patient_data.csv.

    Empty cells in `exercise_frequency` (~2012 rows) and `education_level`
    (~1040 rows) are MISSING DATA - they are not a "None" category. Pandas
    converts the empty cells to NaN; downstream consumers (the preprocessor,
    the predict pipeline, compare_patients) must impute / handle NaN before
    feeding the encoders.

    Note on the saved encoder: it was historically fit with the literal
    string "None" as a fourth category for these columns. That was wrong -
    those rows are missing, not a real category. The encoder still carries
    a "None" slot which is effectively dead because nothing in the pipeline
    feeds it that value anymore; it gets refit cleanly on the next training
    run.
    """
    return pd.read_csv(paths.patient_csv, keep_default_na=False, na_values=[""])


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    patient_csv: Path = REPO_ROOT / "data" / "patient_data.csv"
    clinical_docs_dir: Path = REPO_ROOT / "data" / "documents_data" / "markdowns"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    models_dir: Path = REPO_ROOT / "artifacts" / "models"
    clinical_faiss_dir: Path = REPO_ROOT / "artifacts" / "clinical_faiss_index"
    medical_faiss_dir: Path = REPO_ROOT / "artifacts" / "medical_faiss_index"
    reports_dir: Path = REPO_ROOT / "artifacts" / "reports"
    logs_dir: Path = REPO_ROOT / "artifacts" / "logs"


paths = Paths()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    mlflow_tracking_uri: str = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT}/mlflow.db"
    )
    mlflow_experiment: str = os.environ.get("MLFLOW_EXPERIMENT", "data-doctor")
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
