# hugobarros.cc — Portfolio

Personal portfolio site that bundles several side projects under one deployment.
A single FastAPI process serves a landing page at `/` and mounts each project
as a sub-app on its own URL prefix.

## Projects

| URL | Project | Description |
|---|---|---|
| `/scheduler` | [Scheduler](projects/scheduler/README.md) | Optimization-based weekly scheduling app (FastAPI + FullCalendar + OR-Tools). |
| `/mycompanioncv` | [MyCompanionCV](projects/mycompanioncv/README.md) | AI chatbot that answers as Hugo, grounded in his CV (Gradio + OpenAI). |

## Repo layout

```
.
├── portfolio/              # landing page + top-level FastAPI app
│   ├── app.py              # mounts each project sub-app
│   └── frontend/           # landing.html, i18n.js
├── projects/
│   ├── scheduler/          # self-contained: own backend/, frontend/, README
│   └── mycompanioncv/      # self-contained: own app.py, README (loads from artifacts/mycompanioncv/)
├── artifacts/              # gitignored. Landing-page files at the root;
│                           # per-project assets under artifacts/<project>/
│                           # (e.g. artifacts/mycompanioncv/). Bind-mounted
│                           # into the container in both dev and prod.
├── tests/                  # pytest suite (currently scheduler-only)
├── Dockerfile              # builds the whole portfolio
├── docker-compose.yml      # web + Caddy reverse proxy
├── Caddyfile               # serves hugobarros.cc
└── pyproject.toml          # all deps in one lockfile
```

Each project under `projects/` is self-contained — its code, assets and README
live in its own folder. The only thing shared across projects is the landing
page in `portfolio/`.

## Run locally

```bash
uv sync
uv run uvicorn portfolio.app:app --reload
# → http://localhost:8000
```

The chatbot needs `OPENAI_API_KEY` (and optionally `PUSHOVER_TOKEN` /
`PUSHOVER_USER` for push notifications). Put them in a `.env` file at repo root.

## Run with Docker

**Dev** (live source mounts, port 8000 exposed):
```bash
docker compose up --build
```

**Production** (no source mounts, only Caddy on 80/443, restart + log rotation):
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

In production this is deployed at `hugobarros.cc` (VM `35.231.149.237`),
fronted by Caddy for HTTPS.

## Deploy to production

The prod VM is small (2 vCPU / ~1 GB RAM / 10 GB `pd-standard` disk), so we
**don't build the Caddy image on it** — that step alone took 13+ minutes
because the disk is I/O bound. The custom Caddy image (stock Caddy + the
`caddy-ratelimit` plugin) is built once on a fast machine, pushed to
[ghcr.io/hugobarros96/scheduling-caddy](https://github.com/users/hugobarros96/packages/container/scheduling-caddy)
(public), and pulled on the VM.

### Routine deploy (code changes)

```bash
ssh hugobarros96@35.231.149.237
cd ~/code/web_app_cc
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

`--build` is **required** when code changes — prod removes the source bind
mounts, so the container runs whatever's baked into the `web` image. Without
`--build`, Compose silently reuses the existing image and your changes don't
ship. The Caddy image still comes from ghcr.io as a fast pull (~5 sec); only
the python `web` image rebuilds (~1–2 min, layer cache makes deps reuse).

If nothing in `portfolio/` or `projects/` changed (e.g. you only updated the
README), omit `--build` to save a minute.

### Updating the Caddy image (rare — only when caddy.Dockerfile changes)

Done on your laptop, not the VM:

```bash
# 1. Login to ghcr.io (one-time setup; reuses PAT in .secrets with write:packages scope)
PAT=$(grep -E '^PAT=' .secrets | cut -d= -f2-)
echo "$PAT" | docker login ghcr.io -u hugobarros96 --password-stdin

# 2. Build the image locally (~1 min on a laptop vs 13 min on the VM)
docker build -f caddy.Dockerfile \
  -t ghcr.io/hugobarros96/scheduling-caddy:ratelimit \
  --platform linux/amd64 .

# 3. Push
docker push ghcr.io/hugobarros96/scheduling-caddy:ratelimit
```

Then on the VM, force a refresh (since `pull_policy: missing` won't re-check
the registry when a local copy exists):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull caddy
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Artifacts directory

The `artifacts/` directory (videos, profile photo, CV PDF, etc.) is
gitignored. It must exist on the VM before `up -d`, otherwise the chatbot
will crash at startup trying to read the CV PDF. Sync it from your dev box
when it changes:

```bash
rsync -av --exclude README.md ~/code/scheduling/artifacts/ \
  hugobarros96@35.231.149.237:~/code/web_app_cc/artifacts/
```

After re-syncing, restart only the web service to pick up the new files:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart web
```

## Tests

```bash
uv run pytest
```
