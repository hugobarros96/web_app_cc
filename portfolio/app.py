"""Portfolio top-level FastAPI app.

Serves the landing page at "/" and mounts each sub-project:
  /scheduler/*      → projects.scheduler.backend.app
  /mycompanioncv    → projects.mycompanioncv (Gradio)
"""

from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

from projects.scheduler.backend.app import app as scheduler_app
from projects.mycompanioncv.app import build_demo as build_companion_demo


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="Hugo Barros — Portfolio")


@app.get("/")
async def landing():
    return FileResponse(FRONTEND_DIR / "landing.html")


@app.get("/static/{file_path:path}")
async def landing_static(file_path: str):
    full_path = FRONTEND_DIR / file_path
    if not full_path.is_file():
        return Response(status_code=404)
    return FileResponse(
        full_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Sub-projects
app.mount("/scheduler", scheduler_app)
app = gr.mount_gradio_app(app, build_companion_demo(), path="/mycompanioncv")
