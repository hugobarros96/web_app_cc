"""FastAPI application — serves the API and static frontend."""

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .users import Day, Scheduler, User
from .scheduler import solve

app = FastAPI(title="Scheduling App")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


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


# ── API endpoints ──

@app.post("/api/solve", response_model=SolveResponse)
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


# ── Static files (must be after API routes) ──

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
