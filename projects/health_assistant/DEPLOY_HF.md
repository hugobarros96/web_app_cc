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

# Copy the whole project, including the gitignored artifacts/ (FAISS indices +
# models). It is all self-contained under projects/health_assistant/.
rsync -av --exclude '.git' /home/hbarros/code/scheduling/projects/health_assistant/ ./

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

To run Data Doctor itself locally, build its image from this folder
(`docker build -t datadoctor . && docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... datadoctor`).
The FAISS indices + models live here under `artifacts/`, so it is self-contained.

## Routine deploys (`deploy.sh`)

After the one-time Space setup above, deploy everything from your laptop with the
repo-root script:

```bash
./deploy.sh          # GitHub push + VM redeploy (portfolio, with artifacts) + HF deploy
./deploy.sh vm       # only the portfolio (GitHub push + artifacts rsync + VM redeploy)
./deploy.sh hf       # only deploy Data Doctor to the HF Space
```

The `hf` step is **self-contained to this repo** and pushes everything the Space
needs: it clones with `GIT_LFS_SKIP_SMUDGE=1` (so the ~330 MB of artifacts are
never re-downloaded), then mirrors all of `projects/health_assistant/` (code,
data, and the gitignored `artifacts/` FAISS indices + models) into the Space.
Unchanged artifacts hash to the same git-LFS pointer, so they are not
re-committed or re-uploaded; only changed files are pushed. If your local
`artifacts/` is empty, the step preserves whatever artifacts are already on the
Space rather than deleting them. Tokens (`PAT`, `HF_TOKEN`) are read from
`.secrets`.

After this, the **one-time** clone/copy in step 2 above is only needed for the
very first deploy (to create the Space); routine updates are just `./deploy.sh hf`.
