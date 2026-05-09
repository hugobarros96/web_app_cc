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
│   └── mycompanioncv/      # self-contained: app.py, me/, README
├── tests/                  # pytest suite (scheduler-only at the moment)
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

## Adding a new project
1. Create `projects/<name>/` with its own `app.py` (FastAPI sub-app or Gradio Blocks).
2. Add a `README.md` describing the project.
3. Mount it from `portfolio/app.py` — `app.mount("/<name>", subapp)` for
   FastAPI, `gr.mount_gradio_app(app, demo, path="/<name>")` for Gradio.
4. Add a card to `portfolio/frontend/landing.html` and any new strings to
   `portfolio/frontend/i18n.js`.
5. If new Python deps are needed, add them to top-level `pyproject.toml` and
   run `uv lock`.

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

## Run locally
```bash
uv sync
uv run uvicorn portfolio.app:app --reload
```

## Run with Docker
```bash
docker compose up --build
```

## Deployment
- Docker image, docker-compose with Caddy for HTTPS
- Domain: hugobarros.cc
- Deployment VM IP: 35.231.149.237
