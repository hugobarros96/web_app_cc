# MyCompanionCV

A Gradio chatbot that answers visitor questions as Hugo Barros, grounded in
his CV and a short personal summary. Uses OpenAI tool-calling to:

- record promising leads (`record_user_details`)
- log questions it can't answer (`record_unknown_question`)

Both tools push notifications via Pushover.

Mounted at `/mycompanioncv` in the portfolio app.

## Stack

- **UI** — Gradio `ChatInterface` (mounted into FastAPI)
- **LLM** — OpenAI Chat Completions (`gpt-4o-mini` by default)
- **Grounding** — CV PDF + short summary file; injected into the system prompt
- **Notifications** — Pushover (skipped silently if env vars unset)

## Layout

```
projects/mycompanioncv/
└── app.py        # Me class + tools + build_demo()

artifacts/mycompanioncv/   # gitignored — populated locally and on the deploy VM
├── Curriculum_Vitae_Hugo.pdf
├── summary.txt          # short personal blurb
└── system.txt           # base system prompt
```

`app.py` exposes `build_demo() -> gr.Blocks`; the portfolio app calls this
and mounts the resulting Gradio app under `/mycompanioncv`.

## Environment

Set in a `.env` file at the repo root (loaded via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
PUSHOVER_TOKEN=...        # optional
PUSHOVER_USER=...         # optional
```

If Pushover credentials are missing the `push()` calls become no-ops (the
`record_*` tools still return `{"recorded": "ok"}` so the LLM behavior is
unchanged).

## Run standalone

```bash
uv run python -m projects.mycompanioncv.app
# → opens a Gradio window at http://127.0.0.1:7860
```

## Customizing the persona

Edit `artifacts/mycompanioncv/summary.txt` (free-form blurb) and
`artifacts/mycompanioncv/system.txt` (base system prompt). Replace
`artifacts/mycompanioncv/Curriculum_Vitae_Hugo.pdf` with another CV —
anything the `pypdf` text extractor can read will be appended to the system
prompt. The whole `artifacts/` directory is gitignored.

## Known quirk

The `record_user_details` tool's JSON schema declares `"required": ["email"]`
even though `email` isn't a defined property — preserved as-is from the
upstream version. The OpenAI API tolerates it; tighten the schema if you
ever want strict mode.
