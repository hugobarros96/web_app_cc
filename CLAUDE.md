# Portfolio repo
This is Hugo Barros's personal portfolio — a single Python web app that
bundles several side projects under one deployment. A top-level FastAPI
process serves a landing page at `/` and mounts each project as a sub-app on
its own URL prefix.

## Repo layout
```
.
├── portfolio/              # landing page + top-level FastAPI app
│   ├── app.py              # mounts each project sub-app
│   └── frontend/           # landing.html, i18n.js
├── projects/
│   ├── scheduler/          # self-contained: backend/, frontend/, README
│   ├── mycompanioncv/      # self-contained: app.py, README
│   └── health_assistant/   # Data Doctor: Streamlit + ML. Own Dockerfile;
│                           # deployed to a Hugging Face Space, not mounted.
├── artifacts/              # gitignored. Landing-page files (videos, profile
│                           # photo, links.yaml) sit at the root; per-project
│                           # assets go under artifacts/<project>/
│                           # (e.g. artifacts/mycompanioncv/ holds CV, summary,
│                           # system prompt). Bind-mounted into the container
│                           # in dev and prod.
├── deploy.sh               # one-command deploy (GitHub + VM + HF Space)
├── tests/                  # pytest suite (scheduler + /datadoctor route)
├── Dockerfile, docker-compose.yml, Caddyfile
└── pyproject.toml          # all deps in one lockfile
```

The only thing shared across projects is the landing page in `portfolio/`.
Everything else (backend code, frontend assets, README) lives inside each
project's own folder.

## URL routing
- `/` → portfolio landing page
- `/static/*` → portfolio assets (`i18n.js` for the landing)
- `/scheduler/*` → mounted scheduler sub-app
- `/mycompanioncv` → mounted Gradio chatbot
- `/datadoctor` → HTML page that iframes the Data Doctor Hugging Face Space
  (URL from the `DATADOCTOR_URL` env var; placeholder if unset)

## Adding a new project
1. Create `projects/<name>/` with its own `app.py` (FastAPI sub-app or Gradio Blocks).
2. Add a `README.md` describing the project.
3. Mount it from `portfolio/app.py` — `app.mount("/<name>", subapp)` for
   FastAPI, `gr.mount_gradio_app(app, demo, path="/<name>")` for Gradio.
4. Add a card to `portfolio/frontend/landing.html` and any new strings to
   `portfolio/frontend/i18n.js`.
5. If new Python deps are needed, add them to top-level `pyproject.toml` and
   run `uv lock`.
6. Any private/runtime assets (data files, CVs, demo videos, etc.) go in
   `artifacts/<name>/` — that directory is gitignored and bind-mounted into
   the container. Reference it from the project's `app.py` via
   `Path(__file__).resolve().parents[2] / "artifacts" / "<name>"`.

**Exception for heavy / non-mountable apps** (see Data Doctor below): if a
project cannot mount in-process (e.g. Streamlit) or is too heavy for the 1 GB
VM, do NOT add its deps to the top-level `pyproject.toml` (step 5). Keep it as a
self-contained image with its own Dockerfile, deploy it off-VM (Hugging Face
Space), and add only a small route in `portfolio/app.py` that iframes it.

## Scheduler project
Optimization-based weekly scheduler (FastAPI + FullCalendar + OR-Tools CP-SAT).
See [projects/scheduler/README.md](projects/scheduler/README.md) for the
full feature spec (PT availability, per-user slots, group slots, results
page, etc.).

## MyCompanionCV project
Gradio chatbot that answers as Hugo, grounded in his CV PDF + summary.
See [projects/mycompanioncv/README.md](projects/mycompanioncv/README.md).
Needs `OPENAI_API_KEY` (and optionally `PUSHOVER_TOKEN`/`PUSHOVER_USER`)
in a `.env` file at the repo root.

## Data Doctor project
Clinical-analytics assistant: Streamlit UI over a Strands agent (XGBoost
predictions, sandboxed pandas analytics, hybrid FAISS+BM25 RAG with a
cross-encoder reranker, guardrails, MLflow tracing). See
[projects/health_assistant/README.md](projects/health_assistant/README.md).

It loads torch + transformer models (~1.5 to 2.5 GB RAM), which the 1 GB VM
cannot run, and Streamlit cannot be mounted in-process anyway. So it is NOT a
mounted sub-app: it runs as its own image on a free Hugging Face Space (16 GB),
and `/datadoctor` in `portfolio/app.py` just iframes that Space. Set
`DATADOCTOR_URL` in the repo-root `.env` to the Space URL (it falls back to a
placeholder if unset, which will refuse to frame).

- Code: `projects/health_assistant/` with its own `Dockerfile` + `pyproject.toml`
  (CPU-only torch via the PyTorch CPU index, ~2.3 GB image vs ~6.4 GB CUDA).
- Runtime artifacts (FAISS indices ~322 MB + models) live in
  `projects/health_assistant/artifacts/`, **gitignored** (too big for GitHub) but
  present in the working tree so the project is self-contained. The whole project
  is also excluded from the `web` image via `.dockerignore` (the VM never needs
  it). The HF Space repo (git-LFS) is their versioned home.
- Space secrets: `OPENAI_API_KEY` (required), optional `SERPA_API_KEY`.
- One-time Space setup: [projects/health_assistant/DEPLOY_HF.md](projects/health_assistant/DEPLOY_HF.md).
- Routine sync to the Space: `./deploy.sh hf` (clones with
  `GIT_LFS_SKIP_SMUDGE=1`, mirrors all of `projects/health_assistant/` including
  the gitignored `artifacts/`, pushes only what changed via git-LFS). Fully
  self-contained to this repo, no external source repo needed.
- `./deploy.sh` also rsyncs the gitignored `artifacts/` to the VM (the `vm` step)
  so the landing videos / CV / chatbot summary reach production without a manual
  rsync.

## Run locally
```bash
uv sync
uv run uvicorn portfolio.app:app --reload
```

## Run with Docker
```bash
docker compose up --build
```
## Git
When connecting to git use the PAT directly:
Example:
git push https://PAT@github.com/hugobarros96/web_app_cc.git main
PAT is defined in .secrets - PAT=

## Deployment

- Docker image + docker-compose, Caddy in front of the `web` container for HTTPS.
- Domain: hugobarros.cc — VM IP: 35.231.149.237 (user `hugobarros96`).
- VM is small (2 vCPU, ~1 GB RAM, 10 GB `pd-standard` disk). It is **not** capable
  of compiling Caddy in a reasonable time (`xcaddy build` took 13+ min, disk I/O
  bound). The custom Caddy image (stock Caddy + `caddy-ratelimit`) is built once
  on a fast machine and pulled on the VM.

### Caddy image is hosted on ghcr.io

- Tag: `ghcr.io/hugobarros96/scheduling-caddy:ratelimit` (**public** package).
- `docker-compose.yml` references it via `image:` only — no `build:` block, so
  Compose pulls instead of building.
- `pull_policy: missing` means the VM only re-checks the registry when no local
  copy exists. Use `docker compose pull caddy` to force a refresh when a new
  version has been pushed.
- `caddy.Dockerfile` is kept in the repo as the source of truth for how the image
  is built. Uses BuildKit cache mounts (`/root/.cache/go-build`, `/go/pkg/mod`)
  so local rebuilds reuse Go's module + compile caches.

### Routine deploy (code changes only)

Easiest: `./deploy.sh` from the laptop pushes to GitHub, redeploys the portfolio
on the VM, and syncs Data Doctor's code to the HF Space (`./deploy.sh vm` or
`./deploy.sh hf` for one target). The manual VM flow below is the equivalent of
the `vm` part:

```bash
ssh hugobarros96@35.231.149.237
cd ~/code/web_app_cc
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=30 web
```

The `web` image still builds on the VM (`Dockerfile`, ~1–2 min with cache).
Caddy is just pulled from ghcr.io.

### Updating the Caddy image — runs on the laptop, not the VM

Only needed when `caddy.Dockerfile` actually changes (e.g. bumping the plugin,
adding another `--with`):

```bash
# Auth (one-time; PAT from .secrets, scope write:packages)
PAT=$(grep -E '^PAT=' .secrets | cut -d= -f2-)
echo "$PAT" | docker login ghcr.io -u hugobarros96 --password-stdin

# Build + push (~1 min total on a laptop)
docker build -f caddy.Dockerfile \
  -t ghcr.io/hugobarros96/scheduling-caddy:ratelimit \
  --platform linux/amd64 .
docker push ghcr.io/hugobarros96/scheduling-caddy:ratelimit
```

Then on the VM:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull caddy
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### artifacts/ on the VM

`artifacts/` is gitignored. Without it the web container crashes at startup
(`mycompanioncv` reads the CV PDF at import time). Sync from the dev box:

```bash
rsync -av --exclude README.md ~/code/scheduling/artifacts/ \
  hugobarros96@35.231.149.237:~/code/web_app_cc/artifacts/
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart web
```

### Disk hygiene on the VM

The 10 GB disk fills quickly with docker layers and build cache. If a build
fails on disk space, reclaim with:

```bash
docker builder prune -a -f    # build cache (often the biggest culprit)
docker container prune -f
docker image prune -f
docker network prune -f
df -h /
```

Don't run `docker system prune -a` casually — it would delete the cached
caddy image and force a re-pull (cheap now since it's on ghcr.io) and the
`web_app_cc-web` image (forces a full rebuild on next `up -d --build`).
