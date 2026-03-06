from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class Day(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


# A time window is (day, start_minutes, end_minutes) where minutes are from midnight.
# e.g. Monday 10:00-11:30 -> (Day.MONDAY, 600, 690)
TimeWindow = Tuple[Day, int, int]

SLOT_GRANULARITY_MIN = 15  # smallest block size in minutes
MIN_SLOT_DURATION_MIN = 30
MAX_SLOT_DURATION_MIN = 90
MAX_SLOTS_PER_USER = 4
MIN_SLOTS_PER_USER = 1
MAX_USERS = 50


def minutes_to_hhmm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def hhmm_to_minutes(hhmm: str) -> int:
    parts = hhmm.split(":")
    return int(parts[0]) * 60 + int(parts[1])


@dataclass
class SlotRequest:
    """Represents one session a user needs scheduled.

    duration_min must be between MIN_SLOT_DURATION_MIN and MAX_SLOT_DURATION_MIN
    and a multiple of SLOT_GRANULARITY_MIN.
    """

    duration_min: int

    def __post_init__(self) -> None:
        if self.duration_min < MIN_SLOT_DURATION_MIN:
            raise ValueError(
                f"Slot duration {self.duration_min}min is below minimum "
                f"{MIN_SLOT_DURATION_MIN}min"
            )
        if self.duration_min > MAX_SLOT_DURATION_MIN:
            raise ValueError(
                f"Slot duration {self.duration_min}min exceeds maximum "
                f"{MAX_SLOT_DURATION_MIN}min"
            )
        if self.duration_min % SLOT_GRANULARITY_MIN != 0:
            raise ValueError(
                f"Slot duration must be a multiple of {SLOT_GRANULARITY_MIN}min"
            )

    @property
    def blocks(self) -> int:
        """Number of 30-min blocks this slot occupies."""
        return self.duration_min // SLOT_GRANULARITY_MIN


@dataclass
class User:
    """A regular user who needs to be scheduled into the PT's calendar."""

    name: str
    availability: List[TimeWindow] = field(default_factory=list)
    slot_requests: List[SlotRequest] = field(default_factory=list)
    color: str = "#3B82F6"  # default blue, overridden with random color

    def __post_init__(self) -> None:
        if len(self.slot_requests) > MAX_SLOTS_PER_USER:
            raise ValueError(
                f"User can have at most {MAX_SLOTS_PER_USER} slots"
            )

    def add_availability(self, day: Day, start: str, end: str) -> None:
        """Add an availability window. start/end in 'HH:MM' format."""
        s = hhmm_to_minutes(start)
        e = hhmm_to_minutes(end)
        if s >= e:
            raise ValueError("Start time must be before end time")
        self.availability.append((day, s, e))

    def add_slot_request(self, duration_min: int) -> None:
        if len(self.slot_requests) >= MAX_SLOTS_PER_USER:
            raise ValueError(
                f"User can have at most {MAX_SLOTS_PER_USER} slots"
            )
        self.slot_requests.append(SlotRequest(duration_min=duration_min))

    @property
    def total_blocks_needed(self) -> int:
        return sum(sr.blocks for sr in self.slot_requests)


@dataclass
class Scheduler:
    """The PT / Scheduler user who owns the calendar.

    Manages the pool of regular users and defines the master availability
    within which all sessions must be placed.
    """

    name: str = "Scheduler"
    availability: List[TimeWindow] = field(default_factory=list)
    users: List[User] = field(default_factory=list)

    def add_availability(self, day: Day, start: str, end: str) -> None:
        """Add an availability window. start/end in 'HH:MM' format."""
        s = hhmm_to_minutes(start)
        e = hhmm_to_minutes(end)
        if s >= e:
            raise ValueError("Start time must be before end time")
        self.availability.append((day, s, e))

    def add_user(self, user: User) -> None:
        if len(self.users) >= MAX_USERS:
            raise ValueError(f"Cannot add more than {MAX_USERS} users")
        self.users.append(user)

    def remove_user(self, name: str) -> None:
        self.users = [u for u in self.users if u.name != name]

    def get_user(self, name: str) -> User | None:
        for u in self.users:
            if u.name == name:
                return u
        return None
