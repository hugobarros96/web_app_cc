# syntax=docker/dockerfile:1.6
# CPU-only Data Doctor image for Hugging Face Spaces (Docker SDK).
# torch is pulled from the PyTorch CPU index via pyproject [tool.uv.sources],
# cutting the image from ~6.4 GB (CUDA) to ~2.3 GB.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -e . --link-mode=copy

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY app ./app
COPY data ./data
COPY .streamlit ./.streamlit
COPY pyproject.toml ./
# Bake the local-RAG HuggingFace models (MiniLM + bge-reranker-base, ~370 MB)
# into the image so the first request doesn't hang on a cold download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('BAAI/bge-reranker-base')"
# artifacts/ (FAISS indices + trained models) are committed into the Space repo
# (git-LFS) and present at the build-context root — copy them in.
COPY artifacts ./artifacts
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
