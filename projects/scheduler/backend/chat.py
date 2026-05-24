"""Request/response models for the scheduler chat agent.

Action vocabulary mirrors `docs/agent.md` §5. The agent returns a list of
typed actions and the frontend applies them in order.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Snapshot of frontend state, sent on every request ──

class ChatTimeWindow(BaseModel):
    day: int  # 0=Monday … 6=Sunday
    start: str  # "HH:MM"
    end: str
    event_id: Optional[str] = None  # set so the agent can target remove_availability


class ChatUserSnapshot(BaseModel):
    """A regular user as seen by the frontend."""

    id: str  # frontend-assigned id (e.g. "user_17")
    name: str
    color: str
    slots: List[int]  # durations in minutes
    availability: List[ChatTimeWindow]


class ChatStateSnapshot(BaseModel):
    """The full editable state the agent can mutate, sent each turn."""

    users: List[ChatUserSnapshot]
    pt_availability: List[ChatTimeWindow]


# ── Conversation history ──

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# ── Action vocabulary (matches docs/agent.md §5) ──

class CreateUserPayload(BaseModel):
    name: str
    color: Optional[str] = None
    slots: List[int]
    availability: Optional[List[ChatTimeWindow]] = None
    member_names: Optional[List[str]] = None


class AddAvailabilityPayload(BaseModel):
    user_id: str  # frontend user id, or the literal "pt"
    day: int
    start: str
    end: str


class AddSlotPayload(BaseModel):
    user_id: str
    duration: int


class RemoveAvailabilityPayload(BaseModel):
    user_id: str  # or "pt"
    event_id: str


class ClearAvailabilityPayload(BaseModel):
    user_id: str  # or "pt"


class ChatAction(BaseModel):
    """One mutation the frontend should apply.

    `type` discriminates which payload shape is valid. We keep `payload` as a
    plain dict here so the model stays trivially serializable; the agent is
    expected to build payloads from the typed helpers above.
    """

    type: Literal[
        "create_user",
        "add_availability",
        "add_slot",
        "remove_availability",
        "clear_availability",
    ]
    payload: dict


# ── Request / response envelopes ──

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    state: ChatStateSnapshot


class ChatResponse(BaseModel):
    reply: str
    actions: List[ChatAction] = Field(default_factory=list)
    waiting_for_input: bool = False
