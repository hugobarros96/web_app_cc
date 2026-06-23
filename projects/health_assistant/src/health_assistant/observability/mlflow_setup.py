"""Single-call wiring of MLflow as both experiment tracker AND agent tracer.

Why MLflow as the only observability backend (vs. Phoenix/Langfuse/LangSmith):
MLflow 2.14+ supports `@mlflow.trace`, `mlflow.openai.autolog`, and
`mlflow.langchain.autolog` natively. One UI, one server, one set of credentials.

Production swap: set `MLFLOW_TRACKING_URI` to a personal hosted server
(DagsHub, self-host, or SageMaker-managed MLflow).
"""
from __future__ import annotations

import os

import mlflow

from health_assistant.config import settings


def setup_mlflow() -> str:
    """Initialize tracking URI + experiment + auto-instrumentation.

    Idempotent: safe to call multiple times. Returns the active tracking URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(
        os.environ.get("MLFLOW_EXPERIMENT", settings.mlflow_experiment)
    )

    # Auto-trace LLM calls. We enable both backends; only the one matching the
    # active MODEL_PROVIDER actually fires. Each is best-effort so an SDK/version
    # skew can't break startup.
    try:
        mlflow.openai.autolog()  # OpenAI provider
    except Exception:
        pass
    try:
        mlflow.bedrock.autolog()  # AWS Bedrock provider
    except Exception:
        # bedrock integration unavailable (older mlflow); tool spans + langchain
        # autolog still capture the rest of the trace.
        pass

    # Auto-trace LangChain primitives: FAISS retrievals, HuggingFaceEmbeddings
    try:
        mlflow.langchain.autolog()
    except Exception:
        pass

    return uri
