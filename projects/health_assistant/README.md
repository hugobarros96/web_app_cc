---
title: Data Doctor
emoji: 🩺
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Clinical-analytics assistant (POC). Not for clinical use.
---

# Data Doctor 🩺

POC clinical-analytics assistant.
Predicts patient outcomes, queries the dataset, and answers questions grounded
in clinical documents + medical textbooks.

**Status:** POC. Not for clinical use.

---

## Quick start

### Option A - Docker (one-command demo)
refer to .env.example to configure your .env
```bash
cp .env.example .env                  # configure your env
docker compose --profile setup up bootstrap    # one-time: build the FAISS indices (models ship pre-trained)
docker compose up app mlflow                   # serves Streamlit on :8501, MLflow on :5000
```

- **Streamlit UI** at http://localhost:8501
- **MLflow UI** at http://localhost:5000 (Chat Sessions tab groups traces by Streamlit session ID)

### Option B - Local Python (faster iteration)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # configure your env

# One-time: build the FAISS indices (~5-10 min; models ship pre-trained in the repo)
python -m health_assistant.scripts.bootstrap

# (Optional) retrain the models - not needed, they're committed:
#   python -m health_assistant.models.train_copd --n-trials 30
#   python -m health_assistant.models.train_alt  --n-trials 30

# Two terminals:
streamlit run app/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### No OpenAI key? Use AWS Bedrock

The LLM is the **only** component that needs a cloud key — embeddings (MiniLM),
the reranker (bge), the XGBoost models, and `python_analytics` all run locally.
To replace openai with AWS refer to .env.example to build your .env

Then run exactly as above (`streamlit run …` or `docker compose up app mlflow`).

---

## What it does

A Streamlit chat UI fronts a **Strands agent** with **yped tools**, wrapped by
input/output guardrails and session-scoped memory; the agent routes each turn to the
right tool. Full reference - tools, RAG + cross-encoder reranker, input handling,
guardrails, active learning, observability.

| Question shape | Tool |
|---|---|
| Predictions for a hypothetical patient | `predict_patient_outcomes` |
| Population queries over the 10k-row dataframe | `python_analytics` |
| Facts in the 1,050 patient encounter records | `search_clinical_documents` |
| General medical knowledge | `search_medical_knowledge` |
| Side-by-side comparison of 2–5 patient IDs | `compare_patients` |
| Name and remember a cohort across turns | `save_cohort` |
| Current/recent medical info (sidebar toggle) | `web_search` |

### Input handling

Multiple questions pasted in one message are split into separate turns; the 📎 uploader
takes up to 3 PDFs/images per turn (digital-PDF text inlined; images are OCR-by-LLM only,
never clinical-image interpretation). Details in [CLAUDE.md](CLAUDE.md).

### Active learning

Each prediction has a 🩺 feedback widget; eligible corrections (no imputed features + a
real label) accumulate and trigger a gated background retrain that promotes a model only
if it beats the holdout baseline. Full pipeline in [CLAUDE.md](CLAUDE.md).

### Guardrails

Deterministic input/output filters (PII, injection, scope, disclaimer injection) at the
agent boundary, logged to `artifacts/logs/guardrails.jsonl`, each mapping 1:1 to a Bedrock
Guardrails policy for production. Mapping table in [CLAUDE.md](CLAUDE.md).

### Observability

MLflow is the single backend for both model-training tracking (Optuna trials, SHAP) and
agent/LLM tracing (`@mlflow.trace` per tool + per-turn `chat_turn` spans grouped by
session). See [ARCHITECTURE.md](ARCHITECTURE.md).

---

## EDA findings

The dataset is synthetic: COPD class has no learnable signal (best macro-F1 ≈ 0.25, the
4-class baseline) and ALT ≈ BMI (r = 0.9998), which the deployed regressor leans on to
reach R² ≈ 0.999. Full analysis — and the "extract signal where it exists, stay honest
where it doesn't" story — is in **[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)**.

---

## Repo layout

```
src/health_assistant/
  agent/         # Strands factory + system prompt + model provider + session_state + chat_turn wrapper
  tools/         # 7 typed tools (predict, python_analytics, search_*, compare_patients, save_cohort, web_search)
  models/        # train_copd, train_alt, predict, retrain (feedback loop), feature_schema
  rag/           # chunking, clinical + MedRAG ingestion, hybrid retriever, cross-encoder reranker
  attachments/   # PDF reader (pypdf) + image loader (Pillow) + Attachment dataclass
  feedback/      # feedback log + eligibility counter + validation gate
  guardrails/    # input/output filters + policies + JSONL logger
  analytics/     # sandboxed exec
  observability/ # MLflow setup
  scripts/       # bootstrap orchestrator
app/             # Streamlit UI (only UI-aware module)
notebooks/       # 01_eda.ipynb (EDA + findings)
tests/           # focused suite: model inference, sandbox, agent pipeline, AL pipeline, reranker, attachments
artifacts/       # models/ committed; FAISS indices, SHAP plots, logs, mlflow, feedback/ are gitignored
data/            # patient_data.csv + documents_data/ markdowns
```

---

## Design decisions worth calling out

| # | Decision | Why |
|---|---|---|
| D1 | Strands SDK as the agent framework, OpenAI or Bedrock as the runtime | Strands is model-agnostic; swap the LLM with one env var (`MODEL_PROVIDER`). |
| D2 | One tool that predicts both COPD and ALT, returning both every time | Same feature schema. |
| D3 | One Python tool that handles both compute and charts (no SQL tool) | Cleaner agent + more impressive trace. LLM writes real pandas code. |
| D4 | Two hybrid (dense+sparse+RRF) RAG indices, not one | Clinical-record questions and general-medicine questions need different sources. The agent reasons about which to call. |
| D5 | XGBoost over LightGBM | User preference; equivalent capability. |
| D6 | No isotonic calibration | EDA showed COPD has no signal - calibration is cosmetic. The system prompt explicitly tells the LLM the scores are softmax outputs, not probabilities. |
| D7 | MLflow as the single observability backend | MLflow 3.x covers experiments AND traces AND chat sessions in one server. Simpler than Phoenix/Langfuse/LangSmith. |
| D8 | Mock Bedrock KB + Guardrails locally; document the AWS pieces | Real Bedrock KB has a ~$170/mo OpenSearch Serverless floor; mocking avoids spend while preserving the full architectural story. |

---

## Testing

```bash
pytest -q --ignore=tests/test_agent_smoke.py          # offline suite, ~20 sec
OPENAI_API_KEY=... pytest tests/test_agent_smoke.py   # live tests, ~55s
```

The active-learning pipeline test trains a real model end-to-end with `dry_run=True`
(production files untouched);.

---

## With another week

A few priorities (full list in [v2 spec §13](docs/superpowers/specs/2026-05-31-data-doctor-v2-design.md)):

1. **Real Bedrock Knowledge Base integration** - swap local FAISS for managed KB,
   with a recall@k eval on a held-out Q/A set, a SageMaker endpoint, and
   medical-domain embeddings (`PubMedBERT` / `BGE-M3`) over MiniLM.
2. **Per-user auth** - login (Cognito or similar) so cohorts, attachments, and
   history persist per user instead of dying with the browser session.
3. **Smarter routing as tools grow** - a two-stage router (a cheap classifier
   picks the bucket, a focused call executes) and, past ~15 tools, a split into
   router + specialists (analyst / clinical / predictor). Criteria in `ARCHITECTURE.md`.
4. **Model-ops** - a one-click UI rollback for promoted models (the
   `artifacts/models/archive/<ts>/` snapshots already exist) and a published
   reranker precision@k benchmark vs the FAISS+BM25 baseline.
5. **Richer session memory** - move beyond the single last-cohort / last-patient /
   last-prediction snapshot to a structured ledger that survives the sliding
   window: retain salient facts per turn (e.g. each patient the user *describes*
   and its attributes/medications, not just the last one *predicted*) plus any
   open/unanswered questions, so long sessions don't lose context the window
   trims. Likely a `SummarizingConversationManager` over a small entity store.

Other ideas, briefly: an LLM-as-judge critic that adversarially checks tool outputs namely prediction's disclaimer / SHAP / numbers before display; scanned-PDF OCR (Tesseract or AWS Textract).

---

## Citations

- Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). *Benchmarking
  Retrieval-Augmented Generation for Medicine.* arXiv:2402.13178. - Source
  for the `MedRAG/textbooks` corpus.
- Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). *C-Pack: Packed
  Resources For General Chinese Embeddings.* arXiv:2309.07597. - Source for
  `BAAI/bge-reranker-base`.
---
