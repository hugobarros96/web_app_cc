"""LangGraph chat agent for the scheduler.

Public entry point: `run_agent(req: ChatRequest) -> ChatResponse` (async).
See `docs/agent.md` §7 for the pipeline shape:

    classify_topic ─┬─ scheduling ─→ extract_intent → resolve_user → action
                    └─ off-topic ──→ refuse
"""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..backend.chat import ChatAction, ChatRequest, ChatResponse

load_dotenv(override=True)


# ── Shared agent context ───────────────────────────────────────
SCHEDULER_CONTEXT = """You are part of an assistant that helps a personal trainer (PT) manage a weekly session calendar.

Entities:
- One PT — owns availability windows during which sessions may be placed.
  Referred to as "PT", "scheduler", or "trainer" → canonical id is "pt".
- Many users — each has a name, 1–4 session slots, and zero or more
  availability windows. Names are case-insensitive.

Vocabulary:
- A "slot" is a session DURATION a user needs scheduled (e.g. 30 min).
  It is NOT yet bound to a specific day or time — the solver picks the
  time later. Allowed durations: 30, 45, 60, 75, or 90 minutes.
  Max 4 slots per user.
- An "availability window" is (day, start_time, end_time) during which a
  user or the PT is free to be scheduled.
- Days are integers 0..6 (0=Monday, 6=Sunday). Times are "HH:MM" (24h).

The user edits this calendar by typing natural-language requests in a chat
panel. Your job is to interpret those requests. You do not execute the
changes yourself — you emit structured intents that a downstream component
applies to the calendar.

Every request is stateless: the full conversation history and a snapshot
of the current users + PT availability are provided each turn.
"""


# ── Parsed-intent schema (output of extract_intent_node) ───────
class ParsedAvailability(BaseModel):
    """One availability window parsed out of the user's message."""

    target: str = Field(
        description=(
            "Who this window belongs to. Use the literal 'pt' for the "
            "trainer/scheduler/PT. Otherwise the user's name as it appeared "
            "in the message (case will be normalized downstream)."
        )
    )
    days: list[int] = Field(
        description=(
            "Days the window applies to, as integers 0..6 (0=Monday, "
            "6=Sunday). Expand ranges: 'Mon-Fri' -> [0,1,2,3,4], "
            "'every day' -> [0,1,2,3,4,5,6], 'weekends' -> [5,6]."
        )
    )
    start: str = Field(description="Start time in 'HH:MM' 24-hour format, e.g. '09:00'.")
    end: str = Field(description="End time in 'HH:MM' 24-hour format, exclusive of the end minute.")


class SchedulingIntent(BaseModel):
    """A single scheduling operation parsed from the user's message.

    Fill `operation` first, then populate ONLY the fields relevant to that
    operation. Leave everything else at its default (None / empty list).
    """

    operation: Literal[
        "create_user", "add_availability", "add_slot",
        "remove_availability", "clear_availability",
        "needs_clarification",
    ] = Field(
        description=(
            "What the user wants to do. "
            "'create_user': add a new user (requires user_name and at least one slot_duration). "
            "'add_availability': add availability windows for an existing user or the PT. "
            "'add_slot': add a slot DURATION (e.g. 30 min) to an existing user's roster. "
            "Does NOT need a day or time — those come later when the schedule is solved. "
            "'remove_availability' / 'clear_availability': drop windows. "
            "'needs_clarification': use this when required information is missing or the "
            "request is ambiguous; populate clarification_question."
        )
    )

    user_name: Optional[str] = Field(
        default=None,
        description=(
            "Target user's name, exactly as written by the user. Use 'pt' "
            "(lowercase) for the trainer. Required for create_user, "
            "add_slot, and any *_availability operation."
        ),
    )

    slot_durations: list[int] = Field(
        default_factory=list,
        description=(
            "Slot durations in minutes. Allowed values: 30, 45, 60, 75, 90. "
            "Used by create_user (1-4 entries) and add_slot (exactly one)."
        ),
    )

    availability: list[ParsedAvailability] = Field(
        default_factory=list,
        description=(
            "Availability windows. Used by add_availability and optionally "
            "by create_user. Each entry can cover multiple days via its "
            "`days` list — do not duplicate one entry per day."
        ),
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Question to ask the user back. Set ONLY when "
            "operation='needs_clarification'. Ask for only the missing "
            "piece, not a generic re-prompt. Concise and direct question."
        ),
    )


# ── Classifier schema ──────────────────────────────────────────
class IntentLabel(BaseModel):
    """Single-field schema — forces the LLM to pick one of two values."""
    label: Literal["scheduling", "off-topic"] = Field(
        description="Whether the user message is about scheduling users or off-topic."
    )


# ── Graph state ────────────────────────────────────────────────
class State(BaseModel):
    request: ChatRequest

    intent_kind: Optional[Literal["scheduling", "off-topic"]] = None
    parsed_intent: Optional[SchedulingIntent] = None
    actions: list[ChatAction] = Field(default_factory=list)
    reply: str = ""
    waiting_for_input: bool = False


# ── Node-specific system prompts ───────────────────────────────
_CLASSIFY_SYSTEM = SCHEDULER_CONTEXT + """
Your task: decide whether the user's message is about scheduling
(creating users, adding slots, editing availability) or off-topic.

user can be professor/PT/trainer/scheduler/user. Examples:

- "PT works Mon-Fri 9 to 17" -> scheduling (PT availability)
- "trainer is available Monday 10 to 12" -> scheduling
- "add Maria with a 60-min slot" -> scheduling
- "remove user_1's Tuesday" -> scheduling
- "clear PT's calendar" -> scheduling
- "what's the weather?" -> off-topic
- "tell me a joke" -> off-topic
"""

_EXTRACT_SYSTEM = SCHEDULER_CONTEXT + """
Your task: translate the user's message into a SchedulingIntent.
- Pick exactly one `operation`.
- Fill only fields relevant to that operation.
- If required info is missing (e.g. user name, slot duration for
  create_user), use operation='needs_clarification' and write the
  question into `clarification_question`.
- Resolve day ranges like 'Mon-Fri' into [0,1,2,3,4].
- 'add_slot' takes only a user_name and slot_durations. Never ask for
  a day or time for add_slot — those belong to add_availability.

Examples:
- "create user_2 with a 60-min slot" -> operation=create_user,
  user_name='user_2', slot_durations=[60]. (Name is whatever the user
  wrote — including identifiers like 'user_2', 'u1', 'client_A'.)
- "add a 45-min slot to Bob" -> operation=add_slot, user_name='Bob',
  slot_durations=[45]. (No availability, no clarification.)
- "Maria is free Monday 10-12" -> operation=add_availability,
  user_name='Maria', availability=[{target:'Maria', days:[0],
  start:'10:00', end:'12:00'}].
- "add Maria" -> operation=needs_clarification (no slot_durations given).
"""


_PT_ALIASES = {"pt", "scheduler", "trainer"}


class SchedulingAgent:
    """LangGraph pipeline that turns a chat message into ChatActions."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.classifier = llm.with_structured_output(IntentLabel)
        self.extractor = llm.with_structured_output(SchedulingIntent)
        self.graph = self._build_graph()

    # ── Nodes ──────────────────────────────────────────────
    def classify_topic_node(self, state: State) -> dict:
        result: IntentLabel = self.classifier.invoke([
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(content=state.request.message),
        ])
        return {"intent_kind": result.label}

    def extract_intent_node(self, state: State) -> dict:
        result: SchedulingIntent = self.extractor.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=state.request.message),
        ])
        return {"parsed_intent": result}

    def _clarify(self, intent: SchedulingIntent, question: str) -> dict:
        intent.operation = "needs_clarification"
        intent.clarification_question = question
        return {"parsed_intent": intent}

    def resolve_user_node(self, state: State) -> dict:
        """Resolve intent.user_name to an existing user.id or to 'pt'.

        Pure Python — no LLM. Flips intent to needs_clarification when the
        user can't be found or required create_user fields are missing
        (agent.md §3, §4).
        """
        intent = state.parsed_intent
        op = intent.operation

        if op == "needs_clarification":
            return {"parsed_intent": intent}

        raw_name = (intent.user_name or "").strip()

        if raw_name.lower() in _PT_ALIASES:
            intent.user_name = "pt"
            return {"parsed_intent": intent}

        matched = next(
            (
                u for u in state.request.state.users
                if raw_name and u.name.lower() == raw_name.lower()
            ),
            None,
        )

        if op == "create_user":
            if not raw_name:
                return self._clarify(intent, "What name should the new user have?")
            if not intent.slot_durations:
                return self._clarify(
                    intent,
                    f"What slot durations should {raw_name} have? (30/45/60/75/90 min)",
                )
            if matched is not None:
                intent.user_name = matched.id
            return {"parsed_intent": intent}

        if matched is None:
            return self._clarify(
                intent,
                f"I don't know a user named '{raw_name}'. Should I create one?",
            )
        intent.user_name = matched.id
        return {"parsed_intent": intent}

    def action_node(self, state: State) -> dict:
        intent = state.parsed_intent
        target = intent.user_name

        match intent.operation:
            case "needs_clarification":
                return {
                    "reply": intent.clarification_question or "Could you clarify?",
                    "waiting_for_input": True,
                    "actions": [],
                }

            case "create_user":
                windows = [
                    {"day": day, "start": pa.start, "end": pa.end}
                    for pa in intent.availability
                    for day in pa.days
                ]
                payload = {"name": target, "slots": intent.slot_durations}
                if windows:
                    payload["availability"] = windows
                return {
                    "reply": (
                        f"Created {target} with "
                        f"{len(intent.slot_durations)} slot(s)."
                    ),
                    "waiting_for_input": False,
                    "actions": [ChatAction(type="create_user", payload=payload)],
                }

            case "add_availability":
                actions = [
                    ChatAction(
                        type="add_availability",
                        payload={
                            "user_id": target,
                            "day": day,
                            "start": pa.start,
                            "end": pa.end,
                        },
                    )
                    for pa in intent.availability
                    for day in pa.days
                ]
                return {
                    "reply": (
                        f"Added {len(actions)} availability window(s) to {target}."
                    ),
                    "waiting_for_input": False,
                    "actions": actions,
                }

            case "add_slot":
                duration = intent.slot_durations[0]
                return {
                    "reply": f"Added a {duration}-min slot to {target}.",
                    "waiting_for_input": False,
                    "actions": [
                        ChatAction(
                            type="add_slot",
                            payload={"user_id": target, "duration": duration},
                        )
                    ],
                }

            case "remove_availability":
                if target == "pt":
                    candidates = state.request.state.pt_availability
                else:
                    # resolve_user_node will have set `target` to the user's id;
                    # also accept name for safety / direct callers.
                    matched_user = next(
                        (
                            u for u in state.request.state.users
                            if target and (
                                u.id == target
                                or u.name.lower() == target.lower()
                            )
                        ),
                        None,
                    )
                    candidates = matched_user.availability if matched_user else []

                actions = []
                for pa in intent.availability:
                    for day in pa.days:
                        for w in candidates:
                            if (
                                w.day == day
                                and w.start == pa.start
                                and w.end == pa.end
                                and w.event_id
                            ):
                                actions.append(
                                    ChatAction(
                                        type="remove_availability",
                                        payload={
                                            "user_id": target,
                                            "event_id": w.event_id,
                                        },
                                    )
                                )
                                break
                return {
                    "reply": f"Removed {len(actions)} availability window(s).",
                    "waiting_for_input": False,
                    "actions": actions,
                }

            case "clear_availability":
                return {
                    "reply": f"Cleared availability for {target}.",
                    "waiting_for_input": False,
                    "actions": [
                        ChatAction(
                            type="clear_availability",
                            payload={"user_id": target},
                        )
                    ],
                }

            case _:
                return {
                    "reply": "I'm not sure what you meant — could you rephrase?",
                    "waiting_for_input": True,
                    "actions": [],
                }

    def refuse_node(self, state: State) -> dict:
        return {
            "reply": (
                "I can only help with scheduling — creating users and "
                "editing availability."
            ),
            "actions": [],
            "waiting_for_input": False,
        }

    def topic_route(self, state: State) -> str:
        return state.intent_kind or "off-topic"

    # ── Graph ──────────────────────────────────────────────
    def _build_graph(self):
        gb = StateGraph(State)

        gb.add_node("classify_topic_node", self.classify_topic_node)
        gb.add_node("extract_intent_node", self.extract_intent_node)
        gb.add_node("resolve_user_node", self.resolve_user_node)
        gb.add_node("action_node", self.action_node)
        gb.add_node("refuse_node", self.refuse_node)

        gb.add_edge(START, "classify_topic_node")
        gb.add_conditional_edges(
            "classify_topic_node",
            self.topic_route,
            {"scheduling": "extract_intent_node", "off-topic": "refuse_node"},
        )
        gb.add_edge("extract_intent_node", "resolve_user_node")
        gb.add_edge("resolve_user_node", "action_node")
        gb.add_edge("action_node", END)
        gb.add_edge("refuse_node", END)

        return gb.compile()

    async def run(self, req: ChatRequest) -> ChatResponse:
        result = await self.graph.ainvoke({"request": req})
        # LangGraph 1.x with Pydantic state typically returns the State
        # instance, but be lenient in case a version returns a dict.
        if isinstance(result, dict):
            return ChatResponse(
                reply=result.get("reply", ""),
                actions=result.get("actions", []),
                waiting_for_input=result.get("waiting_for_input", False),
            )
        return ChatResponse(
            reply=result.reply,
            actions=result.actions,
            waiting_for_input=result.waiting_for_input,
        )


# ── Module-level singleton + shim for the FastAPI route ────────
_agent: Optional[SchedulingAgent] = None


def _get_agent() -> SchedulingAgent:
    """Construct the agent lazily so module import doesn't depend on env."""
    global _agent
    if _agent is None:
        _agent = SchedulingAgent()
    return _agent


async def run_agent(req: ChatRequest) -> ChatResponse:
    return await _get_agent().run(req)
