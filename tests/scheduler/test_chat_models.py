"""Pydantic validation tests for the chat request/response models."""

import pytest
from pydantic import ValidationError

from projects.scheduler.backend.chat import (
    ChatAction,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStateSnapshot,
    ChatTimeWindow,
    ChatUserSnapshot,
)


class TestChatMessage:
    def test_user_role(self):
        m = ChatMessage(role="user", content="hi")
        assert m.role == "user"

    def test_assistant_role(self):
        m = ChatMessage(role="assistant", content="hello")
        assert m.role == "assistant"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="system", content="x")


class TestChatAction:
    @pytest.mark.parametrize("kind", [
        "create_user", "add_availability", "add_slot",
        "remove_availability", "clear_availability",
    ])
    def test_valid_types(self, kind):
        a = ChatAction(type=kind, payload={})
        assert a.type == kind

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            ChatAction(type="delete_user", payload={})


class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(
            message="hi",
            state=ChatStateSnapshot(users=[], pt_availability=[]),
        )
        assert req.history == []
        assert req.message == "hi"

    def test_with_history_and_state(self):
        req = ChatRequest(
            message="ok",
            history=[ChatMessage(role="user", content="prev")],
            state=ChatStateSnapshot(
                users=[
                    ChatUserSnapshot(
                        id="u1", name="Alice", color="#fff",
                        slots=[60], availability=[],
                    )
                ],
                pt_availability=[
                    ChatTimeWindow(day=0, start="09:00", end="17:00")
                ],
            ),
        )
        assert len(req.history) == 1
        assert req.state.users[0].name == "Alice"

    def test_missing_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(state=ChatStateSnapshot(users=[], pt_availability=[]))


class TestChatResponse:
    def test_defaults(self):
        r = ChatResponse(reply="hi")
        assert r.actions == []
        assert r.waiting_for_input is False

    def test_with_actions(self):
        r = ChatResponse(
            reply="ok",
            actions=[ChatAction(type="create_user", payload={"name": "x"})],
            waiting_for_input=False,
        )
        assert len(r.actions) == 1


class TestChatTimeWindow:
    def test_optional_event_id(self):
        w = ChatTimeWindow(day=2, start="10:00", end="11:00")
        assert w.event_id is None

    def test_with_event_id(self):
        w = ChatTimeWindow(day=2, start="10:00", end="11:00", event_id="ev_42")
        assert w.event_id == "ev_42"
