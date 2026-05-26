"""Portfolio top-level FastAPI app.

Serves the landing page at "/" and mounts each sub-project:
  /scheduler/*      → projects.scheduler.backend.app
  /mycompanioncv    → projects.mycompanioncv (Gradio)
"""

from pathlib import Path

import gradio as gr
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

from projects.scheduler.backend.app import app as scheduler_app
from projects.mycompanioncv.app import build_demo as build_companion_demo


PORTFOLIO_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PORTFOLIO_DIR / "frontend"
ARTIFACTS_DIR = PORTFOLIO_DIR.parent / "artifacts"
LINKS_FILE = PORTFOLIO_DIR / "links.yaml"

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


@app.get("/artifacts/{file_path:path}")
async def landing_artifacts(file_path: str):
    full_path = (ARTIFACTS_DIR / file_path).resolve()
    if ARTIFACTS_DIR not in full_path.parents or not full_path.is_file():
        return Response(status_code=404)
    return FileResponse(full_path)


@app.get("/api/links")
async def landing_links():
    with open(LINKS_FILE) as f:
        return JSONResponse(yaml.safe_load(f))


# Sub-projects
app.mount("/scheduler", scheduler_app)
app = gr.mount_gradio_app(app, build_companion_demo(), path="/mycompanioncv")
