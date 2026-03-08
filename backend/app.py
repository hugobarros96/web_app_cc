"""FastAPI application — serves the API and static frontend."""

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .users import Day, Scheduler, User
from .scheduler import solve

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Scheduling App")


# ── API models ──

class TimeWindowIn(BaseModel):
    day: int  # 0=Monday … 6=Sunday
    start: str  # "HH:MM"
    end: str

class UserIn(BaseModel):
    name: str
    color: str
    slots: List[int]  # durations in minutes
    availability: List[TimeWindowIn]

class SolveRequest(BaseModel):
    pt_availability: List[TimeWindowIn]
    users: List[UserIn]

class AssignmentOut(BaseModel):
    user_name: str
    day: str
    start_time: str
    end_time: str
    duration_min: int

class SolveResponse(BaseModel):
    assignments: List[AssignmentOut]
    total_requested: int
    total_scheduled: int
    status: str


# ── Solver endpoint ──

@app.post("/scheduler/api/solve", response_model=SolveResponse)
async def api_solve(req: SolveRequest):
    scheduler = Scheduler(name="PT")
    for tw in req.pt_availability:
        scheduler.add_availability(Day(tw.day), tw.start, tw.end)

    for u_in in req.users:
        user = User(name=u_in.name, color=u_in.color)
        for tw in u_in.availability:
            user.add_availability(Day(tw.day), tw.start, tw.end)
        for dur in u_in.slots:
            user.add_slot_request(dur)
        scheduler.add_user(user)

    result = solve(scheduler)

    return SolveResponse(
        assignments=[
            AssignmentOut(
                user_name=a.user_name,
                day=a.day.name.capitalize(),
                start_time=a.start_time,
                end_time=a.end_time,
                duration_min=a.duration_min,
            )
            for a in result.assignments
        ],
        total_requested=result.total_requested,
        total_scheduled=result.total_scheduled,
        status=result.status,
    )


# ── Landing page ──

@app.get("/")
async def landing():
    return FileResponse(FRONTEND_DIR / "landing.html")


# ── Scheduler app under /scheduler ──

@app.get("/scheduler")
async def scheduler_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/scheduler/results")
async def results():
    return FileResponse(FRONTEND_DIR / "results.html")


# ── Static files (with no-cache headers) ──

@app.get("/scheduler/static/{file_path:path}")
async def static_files(file_path: str):
    full_path = FRONTEND_DIR / file_path
    if not full_path.is_file():
        from fastapi.responses import Response
        return Response(status_code=404)
    return FileResponse(
        full_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
