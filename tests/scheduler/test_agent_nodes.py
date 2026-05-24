"""Unit tests for the pure-Python agent nodes.

Covers `resolve_user_node` and `action_node` — the policy layer that runs
without any LLM calls. These tests deliberately avoid invoking the graph
or the classifier/extractor.
"""

from __future__ import annotations

import pytest

from projects.scheduler.agent.scheduling_agent import (
    ParsedAvailability,
    SchedulingAgent,
    SchedulingIntent,
    State,
)
from projects.scheduler.backend.chat import (
    ChatRequest,
    ChatStateSnapshot,
    ChatTimeWindow,
    ChatUserSnapshot,
)


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def agent() -> SchedulingAgent:
    """One agent instance shared across all tests (graph build is cheap)."""
    return SchedulingAgent()


def make_state(
    *,
    intent: SchedulingIntent,
    users: list[ChatUserSnapshot] | None = None,
    pt_availability: list[ChatTimeWindow] | None = None,
) -> State:
    """Build a State with a parsed intent and an optional snapshot."""
    req = ChatRequest(
        message="(unused in pure-python tests)",
        state=ChatStateSnapshot(
            users=users or [],
            pt_availability=pt_availability or [],
        ),
    )
    return State(request=req, parsed_intent=intent)


def make_user_snapshot(
    *,
    id: str,
    name: str,
    availability: list[tuple[int, str, str, str]] | None = None,
) -> ChatUserSnapshot:
    """availability tuples are (day, start, end, event_id)."""
    return ChatUserSnapshot(
        id=id,
        name=name,
        color="#000",
        slots=[60],
        availability=[
            ChatTimeWindow(day=d, start=s, end=e, event_id=ev)
            for d, s, e, ev in (availability or [])
        ],
    )


# ── resolve_user_node ──────────────────────────────────────────

class TestResolveUserNode:
    def test_needs_clarification_passes_through(self, agent):
        intent = SchedulingIntent(
            operation="needs_clarification",
            clarification_question="What?",
        )
        out = agent.resolve_user_node(make_state(intent=intent))
        assert out["parsed_intent"].operation == "needs_clarification"
        assert out["parsed_intent"].clarification_question == "What?"

    @pytest.mark.parametrize("alias", ["pt", "PT", "scheduler", "Scheduler", "trainer", "TRAINER"])
    def test_pt_aliases_normalize_to_pt(self, agent, alias):
        intent = SchedulingIntent(operation="clear_availability", user_name=alias)
        out = agent.resolve_user_node(make_state(intent=intent))
        assert out["parsed_intent"].user_name == "pt"
        assert out["parsed_intent"].operation == "clear_availability"

    def test_existing_user_resolves_to_id(self, agent):
        intent = SchedulingIntent(operation="add_slot", user_name="Maria", slot_durations=[60])
        snapshot = [make_user_snapshot(id="user_42", name="Maria")]
        out = agent.resolve_user_node(make_state(intent=intent, users=snapshot))
        assert out["parsed_intent"].user_name == "user_42"
        assert out["parsed_intent"].operation == "add_slot"

    def test_case_insensitive_match(self, agent):
        intent = SchedulingIntent(operation="add_slot", user_name="MARIA", slot_durations=[30])
        snapshot = [make_user_snapshot(id="user_42", name="Maria")]
        out = agent.resolve_user_node(make_state(intent=intent, users=snapshot))
        assert out["parsed_intent"].user_name == "user_42"

    def test_missing_user_for_non_create_clarifies(self, agent):
        intent = SchedulingIntent(operation="clear_availability", user_name="Bob")
        out = agent.resolve_user_node(make_state(intent=intent, users=[]))
        assert out["parsed_intent"].operation == "needs_clarification"
        assert "Bob" in out["parsed_intent"].clarification_question

    def test_create_user_without_name_clarifies(self, agent):
        intent = SchedulingIntent(operation="create_user", user_name=None, slot_durations=[60])
        out = agent.resolve_user_node(make_state(intent=intent))
        assert out["parsed_intent"].operation == "needs_clarification"
        assert "name" in out["parsed_intent"].clarification_question.lower()

    def test_create_user_without_slots_clarifies(self, agent):
        intent = SchedulingIntent(operation="create_user", user_name="Bob", slot_durations=[])
        out = agent.resolve_user_node(make_state(intent=intent))
        assert out["parsed_intent"].operation == "needs_clarification"
        assert "slot" in out["parsed_intent"].clarification_question.lower()

    def test_create_user_with_new_name_keeps_name(self, agent):
        intent = SchedulingIntent(operation="create_user", user_name="Bob", slot_durations=[60])
        out = agent.resolve_user_node(make_state(intent=intent))
        assert out["parsed_intent"].operation == "create_user"
        assert out["parsed_intent"].user_name == "Bob"

    def test_create_user_with_existing_name_resolves_to_id(self, agent):
        """Per agent.md §4 — existing match becomes an edit, name resolves to id."""
        intent = SchedulingIntent(operation="create_user", user_name="Maria", slot_durations=[30])
        snapshot = [make_user_snapshot(id="user_42", name="Maria")]
        out = agent.resolve_user_node(make_state(intent=intent, users=snapshot))
        assert out["parsed_intent"].user_name == "user_42"


# ── action_node ────────────────────────────────────────────────

class TestActionNodeClarification:
    def test_uses_clarification_question(self, agent):
        intent = SchedulingIntent(
            operation="needs_clarification",
            clarification_question="Which Maria?",
        )
        out = agent.action_node(make_state(intent=intent))
        assert out["waiting_for_input"] is True
        assert out["actions"] == []
        assert out["reply"] == "Which Maria?"

    def test_falls_back_when_question_missing(self, agent):
        intent = SchedulingIntent(operation="needs_clarification")
        out = agent.action_node(make_state(intent=intent))
        assert out["waiting_for_input"] is True
        assert out["actions"] == []
        assert out["reply"]  # non-empty fallback


class TestActionNodeCreateUser:
    def test_emits_create_user_action(self, agent):
        intent = SchedulingIntent(
            operation="create_user",
            user_name="user_2",
            slot_durations=[60],
        )
        out = agent.action_node(make_state(intent=intent))
        assert len(out["actions"]) == 1
        action = out["actions"][0]
        assert action.type == "create_user"
        assert action.payload == {"name": "user_2", "slots": [60]}

    def test_create_user_with_availability_fans_out(self, agent):
        intent = SchedulingIntent(
            operation="create_user",
            user_name="user_2",
            slot_durations=[60],
            availability=[
                ParsedAvailability(target="user_2", days=[0, 1, 2], start="10:00", end="11:00")
            ],
        )
        out = agent.action_node(make_state(intent=intent))
        action = out["actions"][0]
        assert "availability" in action.payload
        assert len(action.payload["availability"]) == 3
        assert {w["day"] for w in action.payload["availability"]} == {0, 1, 2}


class TestActionNodeAddAvailability:
    def test_emits_one_action_per_day(self, agent):
        intent = SchedulingIntent(
            operation="add_availability",
            user_name="user_1",
            availability=[
                ParsedAvailability(target="user_1", days=[0, 1, 2, 3, 4], start="09:00", end="17:00")
            ],
        )
        out = agent.action_node(make_state(intent=intent))
        assert len(out["actions"]) == 5
        assert {a.payload["day"] for a in out["actions"]} == {0, 1, 2, 3, 4}
        for a in out["actions"]:
            assert a.type == "add_availability"
            assert a.payload["user_id"] == "user_1"
            assert a.payload["start"] == "09:00"
            assert a.payload["end"] == "17:00"

    def test_multiple_windows_combine(self, agent):
        intent = SchedulingIntent(
            operation="add_availability",
            user_name="pt",
            availability=[
                ParsedAvailability(target="pt", days=[0, 1], start="09:00", end="12:00"),
                ParsedAvailability(target="pt", days=[2], start="14:00", end="16:00"),
            ],
        )
        out = agent.action_node(make_state(intent=intent))
        assert len(out["actions"]) == 3  # 2 + 1


class TestActionNodeAddSlot:
    def test_emits_single_action(self, agent):
        intent = SchedulingIntent(
            operation="add_slot",
            user_name="user_42",
            slot_durations=[45],
        )
        out = agent.action_node(make_state(intent=intent))
        assert len(out["actions"]) == 1
        action = out["actions"][0]
        assert action.type == "add_slot"
        assert action.payload == {"user_id": "user_42", "duration": 45}


class TestActionNodeRemoveAvailability:
    def test_matches_event_id_from_snapshot(self, agent):
        intent = SchedulingIntent(
            operation="remove_availability",
            user_name="user_42",
            availability=[
                ParsedAvailability(target="user_42", days=[1], start="10:00", end="11:00")
            ],
        )
        snapshot = [make_user_snapshot(
            id="user_42", name="Maria",
            availability=[(1, "10:00", "11:00", "ev_77")],
        )]
        # resolve_user_node would normally swap name → id; emulate that here.
        out = agent.action_node(make_state(intent=intent, users=snapshot))
        assert len(out["actions"]) == 1
        assert out["actions"][0].payload["event_id"] == "ev_77"

    def test_no_match_yields_no_actions(self, agent):
        intent = SchedulingIntent(
            operation="remove_availability",
            user_name="user_42",
            availability=[
                ParsedAvailability(target="user_42", days=[5], start="08:00", end="09:00")
            ],
        )
        snapshot = [make_user_snapshot(
            id="user_42", name="Maria",
            availability=[(1, "10:00", "11:00", "ev_77")],
        )]
        out = agent.action_node(make_state(intent=intent, users=snapshot))
        assert out["actions"] == []

    def test_pt_target_uses_pt_availability(self, agent):
        intent = SchedulingIntent(
            operation="remove_availability",
            user_name="pt",
            availability=[
                ParsedAvailability(target="pt", days=[0], start="09:00", end="17:00")
            ],
        )
        pt_avail = [ChatTimeWindow(day=0, start="09:00", end="17:00", event_id="ev_pt_1")]
        out = agent.action_node(make_state(intent=intent, pt_availability=pt_avail))
        assert len(out["actions"]) == 1
        assert out["actions"][0].payload["user_id"] == "pt"
        assert out["actions"][0].payload["event_id"] == "ev_pt_1"


class TestActionNodeClearAvailability:
    def test_emits_single_action(self, agent):
        intent = SchedulingIntent(operation="clear_availability", user_name="pt")
        out = agent.action_node(make_state(intent=intent))
        assert len(out["actions"]) == 1
        action = out["actions"][0]
        assert action.type == "clear_availability"
        assert action.payload == {"user_id": "pt"}


# ── refuse_node + topic_route ──────────────────────────────────

class TestMiscNodes:
    def test_refuse_node_returns_canned_text(self, agent):
        # State doesn't matter for refuse_node.
        intent = SchedulingIntent(operation="needs_clarification")
        out = agent.refuse_node(make_state(intent=intent))
        assert out["waiting_for_input"] is False
        assert out["actions"] == []
        assert "scheduling" in out["reply"].lower()

    def test_topic_route_scheduling(self, agent):
        intent = SchedulingIntent(operation="needs_clarification")
        state = make_state(intent=intent)
        state.intent_kind = "scheduling"
        assert agent.topic_route(state) == "scheduling"

    def test_topic_route_off_topic(self, agent):
        intent = SchedulingIntent(operation="needs_clarification")
        state = make_state(intent=intent)
        state.intent_kind = "off-topic"
        assert agent.topic_route(state) == "off-topic"

    def test_topic_route_defaults_to_off_topic_when_unset(self, agent):
        intent = SchedulingIntent(operation="needs_clarification")
        state = make_state(intent=intent)
        # intent_kind stays None — should fall back to off-topic.
        assert agent.topic_route(state) == "off-topic"
