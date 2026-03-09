"""Scheduling optimizer using Google OR-Tools CP-SAT solver.

The algorithm:
1. Discretises the week into 30-minute blocks (index 0 = Monday 00:00,
   index 1 = Monday 00:30, …, index 335 = Sunday 23:30).
2. For each user slot-request, enumerates every valid starting block where:
   - the entire duration falls within the PT's availability, AND
   - the entire duration falls within the user's availability.
3. Creates a boolean decision variable for each (slot_request, start_block).
4. Adds constraints:
   a. Each slot request is assigned to exactly one start (or zero if we
      allow partial schedules).
   b. No two assigned sessions overlap (at most one session per block).
5. Maximises the number of scheduled slot-requests.

This is a classic constraint-programming formulation solved efficiently by
the CP-SAT solver even for 50 users x 4 slots each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from .users import (
    Day,
    Scheduler,
    SlotRequest,
    TimeWindow,
    User,
    SLOT_GRANULARITY_MIN,
    minutes_to_hhmm,
)

BLOCKS_PER_DAY = 24 * 60 // SLOT_GRANULARITY_MIN  # 48
BLOCKS_PER_WEEK = 7 * BLOCKS_PER_DAY  # 336


def _block_index(day: Day, minutes: int) -> int:
    """Convert (day, minutes-from-midnight) to a global block index."""
    return day.value * BLOCKS_PER_DAY + minutes // SLOT_GRANULARITY_MIN


def _block_to_day_minutes(block: int) -> Tuple[Day, int]:
    """Convert global block index back to (Day, minutes-from-midnight)."""
    day_idx, slot_in_day = divmod(block, BLOCKS_PER_DAY)
    return Day(day_idx), slot_in_day * SLOT_GRANULARITY_MIN


def _availability_to_blocks(windows: List[TimeWindow]) -> Set[int]:
    """Expand a list of availability windows into a set of block indices."""
    blocks: Set[int] = set()
    for day, start_min, end_min in windows:
        b_start = _block_index(day, start_min)
        b_end = _block_index(day, end_min)
        for b in range(b_start, b_end):
            blocks.add(b)
    return blocks


def _find_valid_starts(
    pt_blocks: Set[int],
    user_blocks: Set[int],
    num_blocks: int,
) -> List[int]:
    """Return all block indices where a session of `num_blocks` consecutive
    blocks fits entirely within both the PT's and user's availability,
    and does not cross a day boundary.
    """
    valid: List[int] = []
    for b in sorted(pt_blocks & user_blocks):
        start_day = b // BLOCKS_PER_DAY
        end_day = (b + num_blocks - 1) // BLOCKS_PER_DAY
        if start_day != end_day:
            continue
        if all((b + offset) in pt_blocks and (b + offset) in user_blocks
               for offset in range(num_blocks)):
            valid.append(b)
    return valid


@dataclass
class Assignment:
    """A single scheduled session in the solution."""

    user_name: str
    slot_idx: int
    day: Day
    start_time: str  # "HH:MM"
    end_time: str  # "HH:MM"
    duration_min: int
    start_block: int


@dataclass
class ScheduleResult:
    """Full output of the optimiser."""

    assignments: List[Assignment]
    total_requested: int
    total_scheduled: int
    status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "ERROR"


def solve(scheduler: Scheduler, time_limit_seconds: float = 100.0,
          excluded_solutions: Optional[List[List[Tuple[str, int, int]]]] = None) -> ScheduleResult:
    """Run the CP-SAT optimiser and return the schedule.

    Args:
        scheduler: The Scheduler object containing PT availability and users.
        time_limit_seconds: Max solver time (default 10s, plenty for 50 users).

    Returns:
        ScheduleResult with the list of assignments.
    """
    model = cp_model.CpModel()

    pt_blocks = _availability_to_blocks(scheduler.availability)
    if not pt_blocks:
        return ScheduleResult([], 0, 0, "INFEASIBLE")

    # Collect all (user, slot_index, slot_request) triples
    slot_vars: List[
        Tuple[User, int, SlotRequest, List[Tuple[int, cp_model.IntVar]]]
    ] = []

    total_requested = 0

    for user in scheduler.users:
        user_blocks = _availability_to_blocks(user.availability)
        for s_idx, slot_req in enumerate(user.slot_requests):
            total_requested += 1
            valid_starts = _find_valid_starts(
                pt_blocks, user_blocks, slot_req.blocks
            )
            start_vars: List[Tuple[int, cp_model.IntVar]] = []
            for start_b in valid_starts:
                var = model.new_bool_var(
                    f"{user.name}_s{s_idx}_b{start_b}"
                )
                start_vars.append((start_b, var))
            slot_vars.append((user, s_idx, slot_req, start_vars))

    # Constraint: each slot request gets at most one start position
    scheduled_indicators: List[cp_model.IntVar] = []
    for user, s_idx, slot_req, start_vars in slot_vars:
        if not start_vars:
            # No valid placement exists for this slot
            continue
        bool_vars = [v for _, v in start_vars]
        model.add_at_most_one(bool_vars)

        # Indicator: 1 if this slot is scheduled
        is_scheduled = model.new_bool_var(f"{user.name}_s{s_idx}_sched")
        model.add(sum(bool_vars) == is_scheduled)
        scheduled_indicators.append(is_scheduled)

    # Constraint: no two slots for the same user on the same day
    from collections import defaultdict
    user_day_vars: Dict[str, Dict[int, List[cp_model.IntVar]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for user, s_idx, slot_req, start_vars in slot_vars:
        for start_b, var in start_vars:
            day_idx = start_b // BLOCKS_PER_DAY
            user_day_vars[user.name][day_idx].append(var)

    for uname, day_map in user_day_vars.items():
        for day_idx, vars_on_day in day_map.items():
            if len(vars_on_day) > 1:
                model.add_at_most_one(vars_on_day)

    # Constraint: no two sessions overlap (at most 1 session per block)
    block_usage: Dict[int, List[cp_model.IntVar]] = {}
    for user, s_idx, slot_req, start_vars in slot_vars:
        for start_b, var in start_vars:
            for offset in range(slot_req.blocks):
                b = start_b + offset
                block_usage.setdefault(b, []).append(var)

    for b, vars_in_block in block_usage.items():
        if len(vars_in_block) > 1:
            model.add_at_most_one(vars_in_block)

    # Objective: maximise number of scheduled slots
    model.maximize(sum(scheduled_indicators))

    # Exclude previous solutions: for each excluded solution, at least one
    # assignment must differ (a "no-good" cut).
    if excluded_solutions:
        # Build lookup: (user_name, slot_idx) -> {start_block: var}
        var_lookup: Dict[Tuple[str, int], Dict[int, cp_model.IntVar]] = {}
        for user, s_idx, slot_req, start_vars in slot_vars:
            var_lookup[(user.name, s_idx)] = {b: v for b, v in start_vars}

        for prev_assignments in excluded_solutions:
            # prev_assignments is a list of (user_name, slot_idx, start_block)
            prev_vars = []
            for uname, s_idx, start_b in prev_assignments:
                key = (uname, s_idx)
                if key in var_lookup and start_b in var_lookup[key]:
                    prev_vars.append(var_lookup[key][start_b])
            if prev_vars:
                # At least one of these must NOT be selected
                model.add(sum(prev_vars) <= len(prev_vars) - 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "ERROR",
        cp_model.UNKNOWN: "ERROR",
    }.get(status, "ERROR")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ScheduleResult([], total_requested, 0, status_name)

    # Extract assignments
    assignments: List[Assignment] = []
    for user, s_idx, slot_req, start_vars in slot_vars:
        for start_b, var in start_vars:
            if solver.value(var):
                day, start_min = _block_to_day_minutes(start_b)
                end_min = start_min + slot_req.duration_min
                assignments.append(
                    Assignment(
                        user_name=user.name,
                        slot_idx=s_idx,
                        day=day,
                        start_time=minutes_to_hhmm(start_min),
                        end_time=minutes_to_hhmm(end_min),
                        duration_min=slot_req.duration_min,
                        start_block=start_b,
                    )
                )

    assignments.sort(key=lambda a: a.start_block)

    return ScheduleResult(
        assignments=assignments,
        total_requested=total_requested,
        total_scheduled=len(assignments),
        status=status_name,
    )
