---
title: Data Doctor
emoji: 🩺
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Clinical-analytics assistant demo. Not for clinical use.
---

# Data Doctor 🩺

A clinical-analytics assistant: a Streamlit chat UI over an LLM **agent** that
predicts patient outcomes, runs live analytics over a 10,000-patient dataset,
and answers questions grounded in clinical records and medical textbooks. The
agent reads each question, picks the right tool, runs it, and explains the
result — all behind input/output safety filters and with full request tracing.

> Demo / portfolio project on synthetic data — **not for clinical use.**

---

## What it does

A **Strands agent** (running on OpenAI, swappable to AWS Bedrock with one env
var) sits behind the chat box. Each turn flows through an input guardrail, the
agent picks one of seven typed tools, the tool runs, and an output guardrail
checks the result before it reaches the UI. Session memory (the last cohort,
patient, and prediction) is carried across turns so follow-ups like *"what if
their BMI were 30?"* resolve implicitly.

| Question shape | Tool |
|---|---|
| Predictions for a hypothetical patient | `predict_patient_outcomes` |
| Population queries over the 10k-row dataframe | `python_analytics` |
| Facts in the 1,050 patient encounter records | `search_clinical_documents` |
| General medical knowledge | `search_medical_knowledge` |
| Side-by-side comparison of 2–5 patient IDs | `compare_patients` |
| Name and remember a cohort across turns | `save_cohort` |
| Current/recent medical info (sidebar toggle) | `web_search` |

- **`predict_patient_outcomes`** — XGBoost models for COPD (4-class, GOLD A–D)
  and ALT (continuous, with an 80% prediction interval). Uses an ask-back
  protocol: it first lists the missing features (ordered by SHAP importance),
  waits for the user, then predicts. COPD scores are presented as relative
  ranks, never as calibrated probabilities.
- **`python_analytics`** — the agent writes real pandas/matplotlib code that
  runs in a sandbox over the patient dataframe; charts are rendered inline.
- **`search_clinical_documents` / `search_medical_knowledge`** — two hybrid-RAG
  indices, one over 1,050 clinical encounter records, one over the
  MedRAG/textbooks corpus (~125k chunks of Harrison's, Robbins, Nelson, etc.).
- **`compare_patients`** — side-by-side feature table + predictions + a bar
  chart for 2–5 patient IDs.
- **`save_cohort`** — names and remembers a cohort from a prior analytics query.
- **`web_search`** — optional, gated by a sidebar toggle; restricted to a
  medical-domain allowlist (CDC, NIH, FDA, WHO, PubMed, Mayo, NEJM, …).

### Input handling

Multiple questions pasted in one message are split into separate turns, each
routed independently. A 📎 uploader takes up to 3 PDFs/images per turn: digital
PDFs are parsed and inlined; images are treated as OCR-by-LLM only (text from
lab printouts or forms) and never as clinical-image interpretation.

### Active learning

Every prediction shows a 🩺 *"Was this prediction correct?"* widget. Eligible
corrections (no imputed features + a real label) accumulate and, past a
threshold, trigger a gated background retrain that only promotes a new model if
it beats the holdout baseline — otherwise production is left untouched.

### Guardrails

Deterministic input/output filters run at the agent boundary: PII redaction,
prompt-injection blocking, a scope check, and disclaimer injection on
prediction / RAG / web answers. Every decision is logged to a JSONL audit file.

### Observability

MLflow is the single backend for both model-training history (Optuna trials,
SHAP) and agent tracing. Each turn is a `chat_turn` span with the OpenAI calls
and every tool nested underneath, grouped by session:

```
[session abc12345]
  ├── chat_turn 1
  │     ├── OpenAI call
  │     ├── search_medical_knowledge
  │     │     ├── FAISS + BM25 retrieval
  │     │     └── reranker.rerank (bge-reranker-base)
  │     └── OpenAI call (final response)
  ├── chat_turn 2
  │     ├── OpenAI call
  │     ├── predict_patient_outcomes
  │     └── OpenAI call (final response)
  └── ...
```

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Streamlit Chat UI  (app/streamlit_app.py)                              │
│  - session_id persisted in URL ?sid=...                                 │
│  - sidebar: web-search toggle | active-learning panel | clear           │
│  - 📎 paperclip uploader (PDF/image, one-shot per turn)                 │
│  - 🩺 inline feedback widgets under every prediction                    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Input Guardrail   │  PII redact, injection
                          │  (deterministic)    │  block, cosine scope
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────▼────────────────────────┐
                  │  Strands Agent (per session_id, OpenAI)   │
                  │  - decision-tree system prompt            │
                  │  - SessionContext injected as prefix      │
                  │    (last cohort, patient, prediction,     │
                  │     named cohorts, web flag)              │
                  │  - attachments threaded as content blocks │
                  │  - 7 typed tools                          │
                  │  - chat_turn parent trace per turn        │
                  └──┬─────┬───── ┬─────┬─────┬─────┬─────┬───┘
                     │     │      │     │     │     │     │
            ┌────────▼┐ ┌──▼────┐ │  ┌──▼────┐ ┌──▼──┐ ┌▼────┐ ┌▼──────┐
            │predict_ │ │python_│ │  │compare│ │save_│ │web_ │ │search_│
            │patient_ │ │analyt.│ │  │_patien│ │cohort│ │searc│ │* (×2) │
            │outcomes │ │(sandb)│ │  │  ts   │ │     │ │  h  │ │       │
            └────┬────┘ └───┬───┘ │  └───┬───┘ └──┬──┘ └──┬──┘ └──┬────┘
                 │          │     │      │        │       │       │
                 │  df ◄────┘     │      │        │  allowlist  ┌─▼──────────────┐
            ┌────▼────────┐       │      │        │   search    │ Hybrid retriever│
            │ XGBoost     │       │      │        │             │ FAISS + BM25    │
            │ COPD + ALT  │       │      │        │             │ + RRF (top-20)  │
            │ + SHAP      │       │      │        │             │       ↓         │
            │ + quantile  │       │      │        │             │ bge-reranker →  │
            │   interval  │       │      │        │             │ top-k           │
            └─────┬───────┘       │      │        │             └─────────────────┘
                  │               │      │        │
                  ▼  ▼  ▼  ▼  ▼  ▼ ▼  ▼  ▼  ▼  ▼ ▼
                          ┌─────────────────┐
                          │ Output Guardrail│  PII rescan, disclaimer
                          │ (deterministic) │  injection, citation check
                          └────────┬────────┘
                                   ▼
                             UI response
```

Each turn calls `agent.run(agent, message, history, session_id, attachments)`,
which returns `{text, tools_used, redactions, flags, figures}`. Session memory
is injected as an ephemeral prompt prefix and scrubbed from stored history each
turn, so only the current turn carries current state. The web-search tool is
added to or removed from the agent's tool registry per turn based on the sidebar
toggle, so when it's off the model literally cannot call it.

### RAG + cross-encoder reranker

Both RAG tools take a 20-doc shortlist per retriever, fuse dense (FAISS) and
sparse (BM25) results with Reciprocal Rank Fusion, then rerank the fused
shortlist with `BAAI/bge-reranker-base` before returning the top-k. Embeddings
are `sentence-transformers/all-MiniLM-L6-v2`. Documents are chunked at 800
tokens / 100 overlap. Both the embedder and the reranker run locally on CPU.

### Models & data

- **15 features** — numeric (age, BMI, medication count, days hospitalized, lab
  glucose, albumin/globulin ratio), binary (readmitted, urban), categorical
  (sex, smoker, diagnosis code), and ordered categorical (exercise frequency,
  diet quality, income bracket, education level).
- **COPD**: XGBoost `multi:softprob` tuned for macro-F1. **ALT**: three XGBoost
  heads (mean + q10/q90 quantiles) giving an 80% interval. Models are committed,
  so predictions work on a fresh clone with no retraining.
- The dataset is **synthetic**. COPD has no learnable signal (macro-F1 ≈ the
  4-class baseline) and ALT tracks BMI almost perfectly — which is why COPD
  outputs are framed as scores, not probabilities, and the app is a
  demonstration of the system, not a clinical model.

---

## Tech stack

- **Agent:** Strands Agents SDK, OpenAI `gpt-4o-mini` (swappable to AWS Bedrock)
- **Models:** XGBoost, SHAP, scikit-learn, Optuna
- **RAG:** sentence-transformers (MiniLM), `bge-reranker-base`, FAISS, BM25,
  LangChain
- **UI:** Streamlit
- **Analytics:** pandas, matplotlib (sandboxed)
- **Observability:** MLflow

---

## Quick start

**Use it:** the app is deployed as a Hugging Face Space and embedded on the
portfolio site at **[hugobarros.cc/datadoctor](https://hugobarros.cc/datadoctor)**.
Try a prompt like *"How many smokers are in the dataset?"* or *"Predict COPD for
a 55-year-old male with BMI 27.5, 3 medications, no exercise, poor diet."*

**Run it locally (Docker):**

```bash
docker build -t datadoctor .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... datadoctor
# → http://localhost:8501
```

The image bakes in the embedding + reranker models; the FAISS indices and
trained models ship with the repo under `artifacts/`. The only thing you need to
supply is `OPENAI_API_KEY` (optionally `SERPA_API_KEY` to enable web search).

---

## Repo layout

```
src/health_assistant/
  agent/         # Strands factory + system prompt + model provider + session state
  tools/         # 7 typed tools
  models/        # train/predict + the feedback-retrain loop + feature schema
  rag/           # chunking, ingestion, hybrid retriever, cross-encoder reranker
  attachments/   # PDF reader + image loader
  feedback/      # feedback log + eligibility counter + validation gate
  guardrails/    # input/output filters + JSONL logger
  analytics/     # sandboxed exec
  observability/ # MLflow setup
  scripts/       # bootstrap (builds the FAISS indices)
app/             # Streamlit UI
data/            # patient_data.csv + clinical document markdowns
artifacts/       # trained models + FAISS indices
```
