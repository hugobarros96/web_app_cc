# Scheduler

Optimization-based weekly scheduler. A "scheduler user" (e.g. a personal
trainer or professor) defines their own weekly availability and a list of
clients/students with their own availability and required slots. The solver
then fits everyone into the trainer's schedule.

Mounted at `/scheduler` in the portfolio app.

## Stack

- **Backend** — FastAPI, OR-Tools (CP-SAT solver)
- **Frontend** — vanilla JS + FullCalendar
- **Optimization** — discretizes the week into 30-min blocks, enumerates
  valid start blocks per slot request, and solves a CP-SAT model that
  maximises scheduled requests subject to no-overlap constraints.

## Layout

```
projects/scheduler/
├── backend/
│   ├── app.py          # FastAPI sub-app — routes are relative
│   │                   # ("/", "/results", "/api/solve", "/static/...")
│   ├── scheduler.py    # CP-SAT solver
│   └── users.py        # Day, User, Scheduler dataclasses
└── frontend/
    ├── index.html      # main calendar UI
    ├── results.html    # results view
    ├── app.js
    ├── style.css
    └── i18n.js         # EN/PT translations
```

## Features

- Up to 50 normal users; each user has 1–4 slots, 30 min – 1h30 each.
- Per-user availability windows (drag-select on the calendar).
- Group slots — a single slot shared across multiple participants.
- Multilingual UI (EN / PT).
- "Generate New Result" — re-solves while excluding previous solutions.

## Run standalone

```bash
uv run uvicorn projects.scheduler.backend.app:app --reload
# → http://localhost:8000
```

In standalone mode, paths are at the root (`/`, `/api/solve`). Through the
portfolio, the same routes live under `/scheduler/*`.

## Tests

```bash
uv run pytest tests/
```
