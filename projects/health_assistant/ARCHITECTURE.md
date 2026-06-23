# Architecture

This document covers the local POC stack (what runs today) and three
production deployment variants on AWS (documented but not built). Every local
component has a 1:1 swap with an AWS service - the architectural story is
intentionally identical at both ends.

---

## Local stack (what `docker compose up` actually runs)

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
                  │    (last cohort, last patient, last       │
                  │     prediction, named cohorts, web flag)  │
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
                 │  df ◄────┘     │      │        │   Serpapi.com │
                 │  + cohort      │      │   session_ctx │     allowlist
                 │  recorder      │   df + predict_both  │        │
                 │                │                      │  ┌────▼────────────┐
            ┌────▼────────┐       │                      │  │ Hybrid retrievr │
            │ XGBoost     │       │                      │  │ FAISS + BM25    │
            │ COPD + ALT  │       │                      │  │ + RRF (top-20)  │
            │ + SHAP      │       │                      │  │       ↓         │
            │ + quantile  │       │                      │  │ bge-reranker-   │
            │ + prediction│       │                      │  │ base → top-k    │
            │   _id       │       │                      │  └─────────────────┘
            └─────┬───────┘       │                      │
                  │               │                      │
                  │  ┌────────────┘                      │
                  │  │                                   │
                  ▼  ▼                                   ▼
            ┌─────────────────┐                  ┌──────────────────┐
            │ Output Guardrail│  PII rescan,     │ Citation check   │
            │ (deterministic) │  disclaimer,     │ extended for web │
            │                 │  image-interp    │ links + domains  │
            └────────┬────────┘  flag, citation  └──────────────────┘
                     │
                     ▼
              UI response  →  user clicks 🩺 → log_feedback() →
                                                  ↓
                              artifacts/feedback/feedback.jsonl
                                                  ↓
                          (5 eligible rows → maybe_trigger_retrain)
                                                  ↓
                              retrain_with_feedback() in daemon thread
                                  ├─ validate (schema / imputation / dup)
                                  ├─ fold + train (best Optuna params)
                                  ├─ evaluate on holdout
                                  └─ if beats floor: archive + os.replace
                                       + reset_model_caches()
```

**Observability** runs alongside: MLflow on `http://localhost:5000` captures
every Optuna trial (per training run, nested), every retrain run (nested
under `feedback_retrain_<N>`), every OpenAI call (via `mlflow.openai.autolog`),
every tool invocation (via `@mlflow.trace`, including the reranker), and every
guardrail decision (via `artifacts/logs/guardrails.jsonl`). The MLflow
**Chat Sessions** tab groups traces by `session_id`, so one Streamlit session
shows as one collapsible thread.

---

## AWS production variants (documented, not built)

Three variants ranked by complexity. The value here is the trade-off
discussion, not picking one "right" answer.

### V1 - Single-task ECS Fargate (recommended starting point)

```
User → ALB → ECS Fargate task (Docker image from ECR)
                ├─→ Bedrock Claude (LLM)
                ├─→ Bedrock Knowledge Base × 2 (RAG)
                ├─→ Bedrock Guardrails (safety)
                ├─→ SageMaker-managed MLflow (observability)
                └─→ S3 (data + model artifacts)
```

- **Why this first**: Streamlit's WebSocket session state works on long-running
  Fargate; not viable on Lambda. Simplest viable cloud deploy. One container,
  one ECS service.
- **Setup**: ECR repo, Terraform/CDK for ECS service + ALB + IAM, GitHub
  Actions for build/push/deploy.
- **Cost floor**: ~$200/mo - Fargate 0.5 vCPU/1 GB always-on (~$15) + ALB
  (~$16) + Bedrock KB on OpenSearch Serverless (~$170, the dominant cost) +
  variable Bedrock invocations + Guardrails (pennies per call).
- **Best for**: single-clinic deployment, internal tool, low concurrency.

### V2 - Decomposed: Fargate UI + Lambda agent + (optional) SageMaker models

```
User → ALB → ECS Fargate (Streamlit only)
                ▼ POST /agent/invoke
            Lambda (Strands agent, 3 GB memory, 60s timeout)
                ├─→ Bedrock Claude / KB × 2 / Guardrails
                └─→ SageMaker Serverless Endpoint (XGBoost models) - optional
```

- **Why**: agent scales to zero between conversations; UI stays warm. Bursty
  workloads pay only for actual usage.
- **Tradeoff**: cold start ~3-8s when loading XGBoost models from S3 on the
  Lambda's first invocation. Mitigations: Provisioned Concurrency (paid),
  or split models to SageMaker Serverless Inference (~$0.20/min when active,
  scales to zero).
- **Cost floor**: ~$190/mo idle (UI Fargate + Bedrock KB floor); scales
  linearly with usage above that.
- **Best for**: bursty workload, multi-tenant SaaS direction.

### V3 - Fully managed: Bedrock Agents + KB + Guardrails

```
User → ALB → ECS Fargate (thin Streamlit client)
                ▼ Bedrock Agent ARN (managed orchestration)
            Bedrock Agents
                ├─ Action group → Lambda (predict, python_analytics)
                ├─ Knowledge Base × 2 (auto-wired)
                └─ Guardrails (auto-applied)
```

- **Why documented**: shows we considered the fully-managed AWS path.
- **Tradeoffs**: trade Strands for AWS's managed agent runtime. Less code, less
  flexibility. **Not recommended** for this POC because Bedrock Agents'
  managed tool-routing loses the custom ask-back UX we designed for
  `predict_patient_outcomes`.
- **Cost floor**: ~$200/mo.

---

## Single-agent vs multi-agent

Data Doctor stays single-agent at 7 tools. Multi-agent adds latency (2x per
turn), token cost, and a harder-to-debug trace tree, so it has to earn its
keep. The thresholds where a split becomes justified:

| Trigger | Multi-agent shape |
|---|---|
| Tool count > ~15 | Router + 2 to 3 specialists (analyst / clinical / predictor) |
| Parallel research workflows | Fan-out + synthesizer |
| Separate-permission boundaries | Privilege-scoped sub-agents |
| High-stakes outputs needing critic | Main + LLM-as-judge critic |

For v2 specifically, the **LLM-as-judge critic over `predict_patient_outcomes`
outputs** is the cheapest defensible multi-agent addition for a clinical
safety check. It's listed in Future Work in the README. The pattern: after
predict returns, a second LLM call re-reads the SHAP features, the disclaimer,
and any numeric claims, and refuses or rewrites the assistant message if it
detects an inconsistency. Adds one LLM hop per prediction turn but composes
cleanly with the existing guardrail pipeline.

---

## Two-stage router (future work)

Today the agent is a single LLM call that does five jobs at once: route the
question to a tool, call it, present the result, enforce safety, and apply
all the special-case rules (markdown tables, attachments, session context,
no LaTeX). One prompt, one context window, one attention budget. When it
mis-routes you cannot tell whether it was a routing error or an execution
error - they share the same trace span.

A two-stage design splits that:

```
                       Stage 1 (cheap classifier)
                       ─────────────────────────────────
  User message ───────▶ gpt-4.1-nano with a structured-
                       output enum:
                         {predict, analytics, clinical_doc,
                          knowledge, compare, save_cohort,
                          web, mixed, unclear}
                       ~100 tokens in/out per call.
                                │
                                ▼
                       Pick the per-bucket prompt + tool subset
                                │
                                ▼
                       Stage 2 (focused execution)
                       ─────────────────────────────────
                       gpt-4.1-mini / 4o with:
                         - a SHORT prompt containing only the
                           rules for that bucket,
                         - ONLY the tools for that bucket
                           (not all 7).
                                │
                                ▼
                       Reply
```

**Wins:**
- Measurable routing accuracy. With a labeled question -> bucket eval set,
  Stage 1 can be scored in isolation. A drop in routing accuracy is
  immediately distinguishable from a regression in answer quality.
- Lower token cost on average. Stage 1 is nano-class; Stage 2 sees a tight
  prompt without the corpus-vs-hypothetical few-shot or attachment rules
  unless they apply.
- Smaller blast radius per prompt edit. You can iterate on the prediction
  prompt without affecting the medical-knowledge prompt.
- Hallucination surface shrinks. Stage 2 only has the tools its bucket
  needs, so it cannot accidentally call a wrong-bucket tool.

**Costs:**
- One extra LLM hop per turn (~0.5-1.0 s of added latency).
- Operational complexity: 1 router prompt + N per-bucket prompts vs 1
  combined prompt.
- Real benefit only materializes once you have an eval harness.

**When to flip the switch:**
- Routing-error rate observed > ~5% on a held-out eval set.
- Tool count crosses ~10 (we're at 7).
- The single prompt becomes too unwieldy to edit confidently.

For v2 the prompt-level investment paid for itself - routing rules now live
in tool docstrings, the corpus-vs-hypothetical case has explicit few-shot,
and multi-question pastes are decomposed upstream in Streamlit before the
agent sees them. Those are the ~80% wins; two-stage routing is the next
~20%, deferred until we either build the eval harness or the tool count
forces it.

---

## Component swap matrix

| Component | Local (POC) | V1 Fargate | V2 Fargate+Lambda | V3 Bedrock Agents |
|---|---|---|---|---|
| LLM | OpenAI gpt-4o-mini | Bedrock Claude | Bedrock Claude | Bedrock Claude |
| Agent runtime | Streamlit process | Container | Lambda | Bedrock Agents |
| RAG | Local FAISS + BM25 | Bedrock KB × 2 | Bedrock KB × 2 | Bedrock KB × 2 |
| Reranker | bge-reranker-base on CPU | Same on container | Same on Lambda (bigger memory) | Bedrock KB reranking (managed) |
| Guardrails | Python filters | Bedrock Guardrails | Bedrock Guardrails | Bedrock Guardrails |
| Models | Joblib in image | Bundled in container | Lambda or SageMaker | Lambda action group |
| Attachments (PDF/image) | In-memory per Streamlit session | S3 upload + per-session prefix | S3 upload + Lambda ingest | S3 + Bedrock Agents file API |
| Web search | Serper.dev + allowlist | Same (Serper API key in Secrets Manager) | Same | Bedrock Agents action group + Serper or Tavily |
| Active learning | feedback.jsonl on disk + background thread retrain | DynamoDB feedback table + SageMaker Pipeline retrain | Same | SageMaker Pipeline triggered by EventBridge |
| Session memory | In-memory `SessionContext` | Same (single Fargate task) | DynamoDB keyed by session_id | Bedrock Agents managed memory |
| Observability | Local MLflow | SageMaker MLflow | SageMaker MLflow | SageMaker MLflow |
| Data | Local CSV + markdowns | S3 | S3 | S3 |
| Cost floor | $0 + OpenAI + Serper | ~$200/mo | ~$190/mo idle | ~$200/mo |
| Cold start | n/a | None | 3-8 s | Managed |

---

## How the swap actually works

Every production swap is a single environment variable:

- `MODEL_PROVIDER=bedrock` + `BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-...`
  → `agent/model_provider.py` returns a `BedrockModel` instead of `OpenAIModel`.
- `MLFLOW_TRACKING_URI=https://...` → MLflow points at the
  SageMaker-managed server. No code change.
- `RAG_BACKEND=bedrock_kb` (planned) → would swap our `HybridRetriever` for a
  thin wrapper around the Bedrock KB `Retrieve` API. The tool interface stays
  the same; only `rag/retriever.py` changes.
- `GUARDRAILS_BACKEND=bedrock` (planned) → would replace the local filter
  functions with calls to Bedrock's `ApplyGuardrail` API. Policies map 1:1.
- `RERANKER_BACKEND=bedrock_kb` (planned) → would skip the in-process
  cross-encoder and let Bedrock KB's managed reranker order the hits. Only
  `rag/reranker.py` changes.
- `WEB_SEARCH_BACKEND=tavily` or `=brave` (planned) → swap the Serper wrapper
  with a different search provider. The allowlist filter and the agent tool
  signature stay the same; only `tools/web_search.py::_serper_results`
  changes.
- `FEEDBACK_BACKEND=dynamodb` + `RETRAIN_TRIGGER=eventbridge` (planned) →
  feedback.jsonl writes become DynamoDB PutItems; the daemon-thread retrain
  becomes a SageMaker Pipeline triggered by an EventBridge rule on N new
  rows. Only `feedback/log.py::log_feedback` and `models/retrain.py`'s
  trigger entry-point change; the validation gate and the training code
  themselves are untouched.

The architectural point: every local-to-cloud swap is a config change at a
single seam, not a code rewrite. That's why the local code is structured as
*filter function* + *interface class* rather than as direct integrations.

---

## CI/CD (documented, not built)

```yaml
on: push to main
  ruff + pytest                                  # fast feedback
  build Docker image (BuildKit cache)            # ~5 min cold, ~30s warm
  tag with git SHA + push to ECR
  terraform apply (drift-check + deploy)
  smoke test the live ALB endpoint

on: data/ changes (separate workflow)
  re-run training notebooks → publish to MLflow
  publish models to S3 only if metrics exceed pinned floors
  cut a new image tag, optional auto-deploy
```

We deliberately didn't ship the workflow YAML in this POC, but the spec
(§10.3) describes the structure.

---

## Observability data model

A single Streamlit chat session produces this MLflow trace tree:

```
[Streamlit session_id = abc12345]
  ├── chat_turn 1   (parent trace, session_id=abc12345, user=streamlit-user)
  │     ├── OpenAI call (auto-instrumented)
  │     ├── search_medical_knowledge   (@mlflow.trace)
  │     │     ├── FAISS retrieval, BM25 retrieval (langchain autolog)
  │     │     └── reranker.rerank      (@mlflow.trace - bge-reranker-base)
  │     └── OpenAI call (final response)
  ├── chat_turn 2   (parent trace, same session_id)
  │     ├── OpenAI call
  │     ├── predict_patient_outcomes   (@mlflow.trace)
  │     └── OpenAI call (final response)
  ├── chat_turn 3   (parent trace, same session_id)
  │     ├── OpenAI call
  │     ├── compare_patients           (@mlflow.trace)
  │     │     └── predict_both × 2 (inline; no separate span)
  │     └── OpenAI call (final response)
  └── ...

[separate parent runs in the same experiment]
  ├── copd_xgb               (30 nested Optuna trials)
  ├── alt_xgb                (30 nested Optuna trials)
  └── feedback_retrain_1     (one nested COPD train + one nested ALT train,
                              tagged with rows_accepted / rows_rejected /
                              delta_macro_f1 / delta_alt_mae / promoted bools)
```

The MLflow Chat Sessions tab groups all `chat_turn` traces with the same
`session_id` into a single collapsible view, so reviewing a conversation is
one click. The training experiments and feedback retrains live in the same
MLflow experiment so the full timeline is visible: initial Optuna
tuning, every accepted-feedback retrain, and the metric deltas at each
promotion.

---

## Tradeoffs we explicitly accepted for the POC

| Choice | Trade-off accepted | Production fix |
|---|---|---|
| Local FAISS + BM25 (in-process) | Single-process; no concurrent writes | Bedrock KB on OpenSearch Serverless |
| AST-guarded `exec()` for python_analytics | POC-grade isolation; can be escaped by a determined adversary | e2b / Modal / gVisor container per call |
| Python regex / cosine for guardrails | Approximate, no contextual grounding | Bedrock Guardrails service |
| Single-tenant Streamlit | No auth, no per-user data isolation | Cognito + per-user IAM-scoped data access |
| One Optuna study per model, no nested CV | Risk of overfitting the hyperparam search | Nested CV + held-out time-based split |
| ALT model leans on BMI (corr ≈ 0.9998) for its R² ≈ 0.999 | On synthetic data we can't tell genuine physiology from a data artifact | Validate the BMI↔ALT relationship on real, non-synthetic data before trusting it |
| Streaming dropped from v2 | Output guardrail can only run on the assembled text, so tokens would render unredacted before the filter could act | Token-level redaction with an inline moderation model |
| Ephemeral session memory + attachments | Lost on browser refresh; bytes for uploaded files are not persisted | Persistent store keyed by authenticated user identity |
| Any-imputation strict reject in feedback validation | Predictions made with even one default-imputed feature can't enter training, even when the user provides correct labels | Per-feature confidence weighting + partial-row updates |
| Cross-encoder reranker runs on CPU per request | ~0.5 to 1.5 s added latency per RAG call; +280 MB Docker image | GPU inference behind a SageMaker endpoint, or Bedrock KB managed reranking |
| Active-learning retrain runs in a Python daemon thread | Single-process; a Streamlit restart loses any in-flight retrain | SageMaker Pipeline triggered by EventBridge on feedback-table writes |
| Web-search allowlist is a hardcoded set | Adding a new trusted domain requires a code change | A managed allowlist in Parameter Store + a periodic review cadence |
