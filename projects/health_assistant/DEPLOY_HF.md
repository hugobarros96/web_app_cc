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
- The portfolio passes `DATADOCTOR_URL` through to the `web` container — add it
  to the `environment:` block in `docker-compose.yml` if it isn't picked up from
  `.env` automatically.
