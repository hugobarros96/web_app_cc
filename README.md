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

## Tests

```bash
uv run pytest
```
