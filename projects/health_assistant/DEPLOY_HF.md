# Deploying Data Doctor to a Hugging Face Space

One-time setup (needs your HF account). Replace `<USER>` with your HF username.

## 1. Create the Space
- huggingface.co → New Space → name `datadoctor`, SDK **Docker**, hardware **CPU basic (free)**.

## 2. Push the code + artifacts (git-LFS)

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/<USER>/datadoctor /tmp/datadoctor-space
cd /tmp/datadoctor-space
git lfs install

# Copy the project files (code, docs, Dockerfile, .streamlit, data, .gitattributes)
rsync -av --exclude artifacts /home/hbarros/code/scheduling/projects/health_assistant/ ./

# Copy the runtime artifacts from the SOURCE repo (FAISS indices + trained models)
mkdir -p artifacts
rsync -av \
  /home/hbarros/code/health_assistant/artifacts/clinical_faiss_index \
  /home/hbarros/code/health_assistant/artifacts/medical_faiss_index \
  /home/hbarros/code/health_assistant/artifacts/models \
  ./artifacts/

git add -A
git commit -m "Deploy Data Doctor"
git push
```

The Space builds the Docker image (~2.3 GB, CPU-only torch, ~10 min first build)
and boots Streamlit on port 8501.

## 3. Set secrets (Space → Settings → Variables and secrets)
- `OPENAI_API_KEY` (required), `OPENAI_MODEL=gpt-4o-mini`, `MODEL_PROVIDER=openai`
- optional: `SERPA_API_KEY` (enables the web_search tool)

## 4. Wire the portfolio
- Set `DATADOCTOR_URL=https://<USER>-datadoctor.hf.space` in the portfolio repo
  root `.env` (read by `portfolio/app.py`). Until set, `/datadoctor` falls back
  to a placeholder URL.
- The portfolio passes `DATADOCTOR_URL` through to the `web` container (already
  wired in `docker-compose.yml`); it's read from the repo-root `.env`.

## Testing locally

The chosen local-test approach: run **only the portfolio** locally and point the
iframe at the **live HF Space** — fast, no need to build the 2.3 GB image.

```bash
# from the repo root
DATADOCTOR_URL=https://hugobarros96-datadoctor.hf.space \
  uv run uvicorn portfolio.app:app --reload
# → http://localhost:8000/datadoctor  iframes the live Space
# → http://localhost:8000/            shows all three project cards
```

To work on Data Doctor itself locally, use its own source repo
(`~/code/health_assistant`, `docker compose up app mlflow`) — that's its full
dev stack with MLflow.

## Routine deploys (`deploy.sh`)

After the one-time Space setup above, deploy everything from your laptop with the
repo-root script:

```bash
./deploy.sh          # GitHub push + VM redeploy (portfolio) + HF code sync
./deploy.sh vm       # only the portfolio (GitHub push + VM redeploy)
./deploy.sh hf       # only sync Data Doctor code to the HF Space
```

The `hf` step syncs **code only** — it clones the Space with
`GIT_LFS_SKIP_SMUDGE=1` (so the 330 MB of artifacts are never re-downloaded),
rsyncs `projects/health_assistant/` over the top excluding `artifacts/`, and
pushes only if code changed. Your committed FAISS indices + models on the Space
are left untouched. Tokens (`PAT`, `HF_TOKEN`) are read from `.secrets`.
