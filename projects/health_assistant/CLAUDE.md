# CLAUDE.md

Working notes for Claude Code / contributors. User-facing setup is in
[README.md](README.md); architecture diagrams and the AWS deployment variants
are in [ARCHITECTURE.md](ARCHITECTURE.md).

## What this is

POC clinical-analytics assistant: a Streamlit chat UI over a **Strands agent**
that predicts patient outcomes (COPD, ALT), queries the 10k-row patient
dataframe, and answers document- and knowledge-based questions. POC only — not
for clinical use.

## Commands

```bash
# One-time setup: build the two FAISS indices (models ship committed under
# artifacts/models/, so bootstrap does NOT train them). Skips an index if it
# already exists; --force rebuilds, --skip-medrag skips the 125k-chunk MedRAG
# index for a fast local setup.
python -m health_assistant.scripts.bootstrap

# Run
streamlit run app/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Test
pytest -q --ignore=tests/test_agent_smoke.py          # offline suite (~50 tests)
OPENAI_API_KEY=... pytest tests/test_agent_smoke.py   # live agent tests

# (Re)train models — not needed normally, they're committed
python -m health_assistant.models.train_copd --n-trials 30
python -m health_assistant.models.train_alt  --n-trials 30

# Docker
docker compose --profile setup up bootstrap
docker compose up app mlflow
```

## Layout

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
tests/           # focused suite
artifacts/       # models/ committed; FAISS indices, logs, mlflow, feedback/ are gitignored
data/            # patient_data.csv + documents_data/ markdowns
```

## Request flow

`user turn → input guardrail → Strands agent (picks a tool) → tool → output
guardrail → UI`. The entry point is `agent.run(agent, message, history, session_id,
attachments)` returning `{text, tools_used, redactions, flags, figures}`; the
factory is `agent.build_agent()` (which also warms the HF embedder + reranker).
Session memory (last cohort, last patient, last prediction, named cohorts) is
injected as a `[session context]` prompt prefix (`agent/session_state.py`,
rendered by `build_context_block()`) so follow-up turns can reference state
implicitly. The block is **ephemeral**: `run()` scrubs it from prior stored
user messages each turn (`_strip_session_context_from_history`) so only the
current turn carries current state — otherwise stale, contradictory snapshots
(old "Last prediction" vs new) accumulate and confuse the model. `Last patient`
is rendered with a caveat that it's the most-recent-*prediction* input and may
not be the patient the user is now asking about. Each turn is wrapped in an
MLflow `chat_turn` trace tagged with `session_id`.

**Concurrency gotchas worth knowing:**
- The whole agent invocation is serialized by a module-level `_AGENT_LOCK`.
  Streamlit reruns can abort a turn mid-flight and leave Strands' internal lock
  set → `ConcurrencyException`; `run()` catches it once, rebuilds a fresh agent,
  and retries.
- **Session state is single-tenant.** `st.session_state["session_ctx"]` is the
  canonical copy on the Streamlit thread; a module-level `_GLOBAL_CTX` + lock
  mirrors it so tool worker threads (where `get_script_run_ctx()` is `None`) can
  read/write the same object. Multi-tenant prod would key this by `session_id`.
- `session_id` is an 8-char UUID persisted in the URL query param `?sid=`, so a
  browser refresh keeps the same MLflow session; the agent is `@st.cache_resource`
  keyed by it.

**History / context-window gotchas:**
- The agent uses a `SlidingWindowConversationManager(window_size=40,
  should_truncate_results=True)` — it trims the OLDEST messages from the front
  of `agent.messages` **in place** each cycle. A long session can therefore trim
  an early free-text patient description before a later follow-up; bump the
  window or switch to `SummarizingConversationManager` if longer recall is
  needed.
- **`tools_used` must be diffed by message-object identity, not index.** Because
  the window manager trims from the front mid-turn, an index slice
  (`agent.messages[len_before:]`) silently misses this turn's tool calls once
  trimming kicks in → `tools_used == []`, which drops the disclaimer, the
  Tools-used panel, AND the feedback widget. `_extract_tools_used_from_messages`
  snapshots `{id(m) …}` before the turn and collects `toolUse` from messages not
  in that set.

## The seven tools

Six are always-on; `web_search` is gated by a sidebar toggle. Gating is
**registration-based**, not prompt-based: when the toggle is OFF the tool is
not registered on the agent at all (the model literally cannot call it); when
ON it is added back. `agent.run()` calls `set_web_search_enabled(agent, ...)`
each turn to sync this in place (no agent rebuild — Strands rebuilds the tool
list from the registry every turn). The in-tool `disabled`-guard stays as
defense in depth.

| Question shape | Tool |
|---|---|
| Predictions for a hypothetical patient | `predict_patient_outcomes` |
| Population queries over the 10k-row dataframe | `python_analytics` |
| Facts in the 1,050 patient encounter records | `search_clinical_documents` |
| General medical knowledge | `search_medical_knowledge` |
| Side-by-side comparison of 2–5 patient IDs | `compare_patients` |
| Name and remember a cohort across turns | `save_cohort` |
| Current/recent medical info (toggle) | `web_search` |

1. **`predict_patient_outcomes(features, ask_back=True, assumptions=None)`** —
   XGBoost models for **COPD** (4-class, GOLD A/B/C/D) and **ALT** (continuous,
   with an 80% prediction interval from quantile heads). Ask-back protocol:
   first call returns `{"status": "needs_input", "missing": [...]}` listing
   missing features ordered by combined SHAP importance, waits for user input or
   "go ahead", then re-calls with imputation (`status": "ok"`). Every successful
   prediction returns a 12-hex `prediction_id` the UI uses to key the feedback
   widgets, and **updates `last_patient` + `last_prediction` in session state**
   so follow-up turns ("what if their BMI were 30?") resolve implicitly.
   **Gotcha:** COPD `class_scores` are raw XGBoost softmax, *not* calibrated
   probabilities — the prompt forces "ranks highest with score X", never "X%
   chance". `assumptions` is a passthrough list (NL→feature mappings the agent
   made) echoed back in the response for the audit trail.
2. **`python_analytics(code, timeout_seconds=30)`** — sandboxed pandas /
   matplotlib over the patient dataframe (`df`, lru-cached). The agent writes
   real code; the trace shows it. AST-guarded exec, no imports, 30-sec SIGALRM
   timeout (main thread only). Figures from `plt.show()` are captured as base64
   PNG into a per-turn registry but **stripped from the dict returned to the
   agent** (token control) — the agent references the chart in words; the UI
   inlines it. If the final expression is a DataFrame with a `patient_id` column
   and 0–5000 rows, it's auto-recorded as the session's "last cohort" (enables
   implicit `save_cohort`).
3. **`search_clinical_documents(query, k=5)`** — hybrid retrieval (FAISS dense +
   BM25 sparse + Reciprocal Rank Fusion) over 1,050 markdown clinical encounters,
   then reranked by a cross-encoder.
4. **`search_medical_knowledge(query, k=5)`** — hybrid retrieval over
   **MedRAG/textbooks** (Xiong et al. 2024, arXiv:2402.13178) — ~125k chunks from
   Harrison's Internal Medicine, Robbins Pathology, Nelson Pediatrics, etc. Also
   reranked. Routing gotcha (in the system prompt): "that patient" / "the heart
   attack patient" means a *hypothetical from earlier in the chat*, NOT a corpus
   record — only search the corpus when the user explicitly points at it.
5. **`compare_patients(patient_ids, include_predictions=True)`** — look up 2–5
   patients by `patient_id` (P00000…), return a side-by-side feature table +
   optional COPD + ALT predictions + an inline bar chart of key numeric features.
   Predictions are computed *inside* the tool per patient (the agent must not
   loop `predict_patient_outcomes`); missing categoricals are imputed via
   population mode/median first to avoid encoder crashes.
6. **`save_cohort(name)`** — persist the most recent `python_analytics` cohort
   (from `last_cohort`) under a name; later turns reference it by name.
   Session-only. Returns `status` ∈ {`saved`, `no_recent_cohort`, `error`}; on
   `no_recent_cohort` the agent must tell the user to run an analytics query
   first, not retry.
7. **`web_search(query, k=5, sites=None)`** *(conditional)* — **SerpApi
   (serpapi.com, via langchain `SerpAPIWrapper`)** Google search — *not* Serper.dev
   despite the legacy `_serper_results` function name and `SERPA_API_KEY` env var.
   Restricted to a ~22-domain medical allowlist (CDC, NIH, FDA, WHO, PubMed, Mayo,
   NEJM, …); the allowlist is enforced both as a `site:` bias (capped at 10 domains)
   *and* as a post-filter on results. Off by default. Gating is
   **registration-based**: `agent.run()` calls `set_web_search_enabled(agent,
   ctx.web_search_enabled)` each turn, adding/removing the tool from the Strands
   tool registry — so when the toggle is OFF the model has no `web_search` tool
   to call (not merely a prompt instruction telling it to abstain). The in-tool
   `SessionContext.web_search_enabled` check remains as a defense-in-depth
   fallback (returns `status="disabled"`). Reads `SERPA_API_KEY` or
   `SERPAPI_API_KEY`. Returns `status` ∈ {`disabled`, `ok`, `no_allowed_results`,
   `error`} — on `error` the agent reports a provider problem rather than retrying.

## RAG + cross-encoder reranker

Both RAG tools (`rag/retriever.py`) take a per-retriever shortlist of
`SHORTLIST_K = 20`, fuse dense + sparse with a LangChain `EnsembleRetriever`
(weights 0.5/0.5, RRF), then route the fused shortlist through
`BAAI/bge-reranker-base` (Xiao et al. 2023, arXiv:2309.07597) before returning
the top-k. Adds ~0.5–1.5s per RAG call on CPU; reranker downloads ~280 MB on
first use (cached under `~/.cache/huggingface/`, baked into the Docker image).

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (`EMBEDDING_MODEL`).
- **Chunking** (`rag/chunking.py`): 800 tokens / 100 overlap. Clinical docs are
  split on markdown H1/H2 then recursively as a safety net; MedRAG keeps the
  publisher's paragraph chunks as-is (no re-chunking).
- **Indices** (gitignored, built by bootstrap): `artifacts/clinical_faiss_index/`
  and `artifacts/medical_faiss_index/`, each `index.faiss` + `index.pkl`.

## Input handling: multi-question pastes + attachments

- **Multi-question pastes** (`app/streamlit_app.py`) — if a single submit contains
  multiple questions, Streamlit splits them into separate queue entries, each its
  own turn with its own tool routing (more reliable than asking the LLM to handle
  N concatenated questions). The splitter tries ordered heuristics, first one that
  yields >1 part wins: (1) blank-line paragraph breaks, (2) numbered lists
  (`1.`/`2)`), (3) split after `?` when the next char is a Latin capital. Chat-export
  stamps like `[19:48, 02/06/2026] Name:` are normalized to paragraph breaks first.
  Attachments travel with the **first** sub-question only.
- **Attachments (📎)** — up to 3 files/turn (max 8 MB each, PDF + PNG/JPG/JPEG/WebP),
  attached to one next prompt only; bytes are not persisted in history (replay shows
  name + badge). Digital PDFs are parsed with `pypdf` and inlined between
  `[Attached PDF: <name>] … [End PDF]` markers (capped 32K chars); pages with text
  <100 chars are flagged as likely-scanned (OCR is future work). Images are
  normalized to RGB PNG, resized to ≤2048 px, and treated as **OCR-by-LLM only** —
  may extract text from lab printouts / labels / forms, must NOT interpret
  X-rays/MRI/CT/ECG. (A former `image_interpretation_attempt` output flag was
  removed — too many false positives; clinical safety now rests on the system
  prompt + disclaimer instead.)

## Guardrails

Deterministic input/output filters at the agent boundary. Minimal for the POC;
each maps 1:1 to a Bedrock Guardrails policy for the production swap. Every
decision is appended to `artifacts/logs/guardrails.jsonl`.

| Local filter | Bedrock Guardrail policy |
|---|---|
| PII regex on input + output (email, phone, SSN → `[PII-REDACTED:*]`) | Sensitive Information Filters |
| Injection regex patterns ("ignore previous instructions", "print your system prompt", "you are now", …) | Custom word filters |
| Scope-vector cosine check | Denied topics |
| Disclaimer injection on prediction / RAG / web responses | Content policy (response template) |

Implementation notes:
- **Scope check** (`input_filter.py`) embeds the input with a *separate*
  **multilingual** model (`paraphrase-multilingual-MiniLM-L12-v2`, so PT/ES/FR
  questions are scored fairly — distinct from the RAG embedder) and compares
  cosine vs the mean of 10 seed queries. Threshold is a deliberately permissive
  **0.05** (POC demo of the Bedrock-Guardrails seam, not real gating); messages
  under 4 words skip the check.
- **Disclaimer injection** fires only for `predict_patient_outcomes`,
  `search_clinical_documents`, `search_medical_knowledge`, `web_search`, and is
  suppressed if the response already contains "analytical aid".
- Every decision appends a row to `artifacts/logs/guardrails.jsonl`:
  `{ts, stage: input|output, policy: pii|injection|scope|disclaimer,
  decision: pass|redact|block|inject|refuse, …extra}`.

Future work: client-specific rules (terminology, org-policy keywords, contextual
grounding, specialty denied-topics) layer on top at deploy time in Bedrock Guardrails.

## Active learning

Each prediction shows a 🩺 *"Was this prediction correct?"* widget that writes a
row to `artifacts/feedback/feedback.jsonl` (full feature dict, predicted values,
user actuals, `kind` ∈ {`thumbs_up`, `correction`}, `eligible_for_training`,
`consumed_by_retrain`, `validation_status`). A row is eligible iff (a) **zero
features were imputed** at prediction time AND (b) at least one actual label was
given. When `RETRAIN_THRESHOLD = 5` eligible-pending rows accumulate, a background
daemon thread runs `retrain_with_feedback()` (guarded by a
`artifacts/feedback/.retrain.lock` sentinel — a concurrent retrain returns
`skipped_in_progress`):

1. **Validation gate** — three sub-gates, all must pass: schema (15 features
   present, numeric clinical ranges e.g. age 0–120 / bmi 10–60, valid categories,
   ≥1 label), strict "no imputed features", and duplicate detection (SHA256 of the
   normalized feature dict) vs the training set + prior accepted rows. A row needs
   ≥**3** *accepted* rows after this gate to proceed (5 is the trigger, 3 is the
   floor to actually train).
2. **Fold + train** reusing the *existing fitted preprocessor* and the initial
   Optuna best params (no re-tuning; ~10–30s/model).
3. **Holdout evaluation** on the same deterministic 80/20 split (`seed=42`) as
   the baseline.
4. **Promotion gate** (independent per model) — COPD: `macro_f1_holdout ≥
   production − 0.005`; ALT: `mae_holdout ≤ production + 0.05`.
5. **Atomic swap** (`os.replace`) if promoted; prior model archived to
   `artifacts/models/archive/<UTC-ts>/`; `predict.py` caches invalidated via
   `reset_model_caches()`.

A retrain that promotes nothing leaves production untouched. `dry_run=True`
(used in tests) runs the full pipeline but never touches production files or
caches. The sidebar shows pending/eligible counters, rejection breakdown, and
the latest retrain delta (COPD macro-F1, ALT MAE).

## Observability

MLflow is the single backend for both model-training tracking (30 Optuna trials
per model, nested runs, SHAP artifacts) and agent/LLM tracing
(`mlflow.openai.autolog` / `mlflow.bedrock.autolog`, `@mlflow.trace` per tool,
per-turn `chat_turn` spans grouped by `session_id` in the Chat Sessions tab).

## Models & data

- **15 features** (`models/feature_schema.py`): numeric — `age`, `bmi`,
  `medication_count`, `days_hospitalized`, `last_lab_glucose`,
  `albumin_globulin_ratio`; binary — `readmitted`, `urban`; categorical —
  `sex`, `smoker`, `diagnosis_code` (D1–D5); ordered categorical —
  `exercise_frequency`, `diet_quality`, `income_bracket`, `education_level`.
  `exercise_frequency` and `education_level` have genuine missing data (imputed
  at predict time), not a "None" category.
- Targets `chronic_obstructive_pulmonary_disease` (4-class GOLD A–D) and
  `alanine_aminotransferase` (regression).
- **COPD**: XGBoost `multi:softprob`, tuned for macro-F1. **ALT**: three XGBoost
  heads — mean (`reg:squarederror`) + q10/q90 quantile heads
  (`reg:quantileerror`) giving the 80% interval — saved as `alt_xgb.joblib`,
  `alt_xgb_q10.joblib`, `alt_xgb_q90.joblib`. Models are **committed** under
  `artifacts/models/`, so predictions work on a fresh clone without retraining.
- `models/predict.py` lru-caches all artifacts (`maxsize=1`) and caches the SHAP
  ask-back feature ordering to `feature_importance_order.json` for reproducibility
  across Streamlit reloads. (Legacy quirk: the COPD label encoder carries a dead
  "None" slot from earlier training; harmless, cleaned up on the next retrain.)
- EDA (synthetic data): COPD has no learnable signal (macro-F1 ≈ 0.25 baseline);
  ALT ≈ BMI (r = 0.9998), which the deployed regressor uses to reach R² ≈ 0.999.
  Full analysis in `notebooks/01_eda.ipynb`.

## LLM provider swap (OpenAI ↔ Bedrock)

- `agent/model_provider.py::get_model()` reads `MODEL_PROVIDER` (`openai` | `bedrock`).
  Only the **chat LLM** swaps; embeddings (MiniLM) and the reranker (bge) always
  run locally, so RAG/predictions need no cloud key.
- Bedrock: set `BEDROCK_MODEL_ID` (use the geo-prefixed cross-region inference
  profile, e.g. `eu.amazon.nova-2-lite-v1:0`), `BEDROCK_REGION`, and credentials
  (`AWS_BEARER_TOKEN_BEDROCK`, or AWS keys, or `AWS_PROFILE`). See `.env.example`.

## Conventions / gotchas

- **Config**: only `.env` is read at runtime (`config.py` load_dotenv); `.env.example`
  is a committed template, never loaded.
- **Docker HF models**: baked into the image (`Dockerfile`, `ARG BAKE_LOCAL_MODELS=true`)
  so the container doesn't download ~370 MB on first request and hang. Build with
  `--build-arg BAKE_LOCAL_MODELS=false` for an AWS-managed-RAG image.
- **Streamlit file-watcher** is disabled (`.streamlit/config.toml` +
  `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none`) — it trips transformers' lazy
  torchvision import. Rerun the app manually after editing source.
- **`python_analytics`** runs AST-guarded `exec` with no imports (`analytics/sandbox.py`).

## Testing notes

The offline suite covers model inference, sandbox safety, the agent pipeline,
the reranker, attachments, web search, and the full active-learning pipeline
(`test_retrain_with_feedback_dry_run_runs_end_to_end` trains a real model
end-to-end with `dry_run=True`, so production files are never touched).
Deliberately skipped: chunking, guardrail PII regexes, tool wrappers — exercised
via live tests + manual demo.
