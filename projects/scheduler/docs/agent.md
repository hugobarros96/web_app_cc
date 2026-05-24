# Scheduler chat agent — requirements

> Single source of truth for the natural-language chat agent that lives at
> the bottom of the scheduler sidebar. The UI, the request/response
> contract, and the route stub are already scaffolded; this document
> describes the agent that should sit behind `POST /scheduler/api/chat`.

## 1. Purpose

Let users edit users, slot durations, and availability with natural
language instead of clicking through dialogs. Typical exchange:

> **User:** *"user_1 is free every day from 17 to 18"*
> **Agent:** *"Got it — added Mon–Sun 17:00–18:00 to user_1."*
> *(returns `add_availability` actions for each weekday)*

> **User:** *"add Maria"*
> **Agent:** *"What slot durations should Maria have? (30/45/60/75/90 min)"*
> *(`waiting_for_input=true`, no actions)*

> **User:** *"60 minutes"*
> **Agent:** *"Created Maria with a 60-min slot."*
> *(returns `create_user` action with `slots=[60]`)*

## 2. Worked examples

Each row is one user turn → expected `ChatResponse` contents.

| User says | `reply` (paraphrased) | `actions` |
| --- | --- | --- |
| `"user_1 is free every day from 17 to 18"` (user_1 exists) | "Added Mon–Sun 17:00–18:00 to user_1." | 7 × `add_availability` (Mon–Sun, 17:00–18:00) |
| `"create user_2 with a 60-min slot"` | "Created user_2." | `create_user` with `{name: "user_2", slots: [60]}` |
| `"add Maria"` | "What slot durations? (30/45/60/75/90)" | `[]`, `waiting_for_input=true` |
| `"60 and 45"` (continuing the previous turn) | "Created Maria with 60 and 45 min slots." | `create_user` with `{name: "Maria", slots: [60, 45]}` |
| `"PT works Mon–Fri 9–17"` | "Set PT availability." | 5 × `add_availability` with `user_id="pt"` |
| `"clear PT's calendar"` | "Cleared PT availability." | `clear_availability` with `{user_id: "pt"}` |
| `"remove Maria's Tuesday slot"` | one of: a question to disambiguate, or a `remove_availability` action | depends |
| `"what's the weather?"` | "I can only help with scheduling — creating users and editing availability." | `[]` |

## 3. User-creation requirements

To emit a `create_user` action the agent must have collected:

- **Name** — required, free-form string. Resolve case-insensitively against
  `state.users`; if a match exists, edit that user instead of creating.
- **At least one slot duration** — one of `{30, 45, 60, 75, 90}` minutes
  (multiples of 15 between 30 and 90, max 4 per user — see
  [`projects/scheduler/backend/users.py`](../backend/users.py)).
- **Availability** — optional at creation time. May be added in the same
  or a later turn.

If any required field is missing, set `waiting_for_input=true`, ask only
for what's missing in `reply`, and emit no `actions`. The next user turn
arrives with the full `history`, so the agent can resume from context — it
does **not** keep server-side memory.

## 4. Existing-user resolution

- Match by `name` case-insensitively against `state.users[].name`.
- If no match: create the user (after collecting required fields).
- "PT" / "scheduler" / "trainer" → the special `pt` target
  (`user_id: "pt"` in action payloads).
- If multiple matches, ask the user to disambiguate.

## 5. Action vocabulary

Mirrors `ChatAction` in
[`projects/scheduler/backend/chat.py`](../backend/chat.py). Frontend
applies actions in order before re-rendering. Day numbers follow the
`Day` enum (0 = Monday … 6 = Sunday). Times are `"HH:MM"` strings.

| `type` | `payload` |
| --- | --- |
| `create_user` | `{ name: str, color?: str, slots: list[int], availability?: list[{day, start, end}], member_names?: list[str] }` |
| `add_availability` | `{ user_id: str \| "pt", day: int, start: str, end: str }` |
| `add_slot` | `{ user_id: str, duration: int }` |
| `remove_availability` | `{ user_id: str \| "pt", event_id: str }` |
| `clear_availability` | `{ user_id: str \| "pt" }` |

For day ranges like "every day" or "Mon–Fri", emit one
`add_availability` per day so the frontend can merge them with its
existing `mergeAvailability()` helper.

## 6. Guardrails

The agent must refuse anything that isn't:

- creating, renaming, or deleting users,
- adding, removing, or clearing availability,
- adding or removing slot durations,
- asking clarifying questions about the above.

Implement guardrails as a **classifier node that runs before any
tool-calling node**, so off-topic input never reaches the
parameter-extraction LLM call. Suggested cheap classifier: a single
LLM call with a few-shot prompt and `temperature=0`, returning
`scheduling | off-topic`. For refusals, return:

```json
{
  "reply": "I can only help with scheduling — creating users and editing availability.",
  "actions": [],
  "waiting_for_input": false
}
```

Additional safety rails:

- **Tool-call budget** — stop after at most 3 tool calls per request.
- **No free-form code execution.** Actions must conform to the schema.
- **No PII echo.** Don't repeat secrets/keys that may appear in input.

## 7. Suggested LangGraph shape

```
                ┌──────────────────┐
                │ classify_intent  │
                └────────┬─────────┘
                         │
                  off-topic │ scheduling
                  ┌──────┘ └────────┐
                  ▼                 ▼
           ┌──────────┐       ┌────────────────┐
           │  refuse  │       │ extract_params │
           └──────────┘       └────────┬───────┘
                                       ▼
                              ┌─────────────────┐
                              │  resolve_user   │
                              └────────┬────────┘
                                       ▼
                              ┌─────────────────┐
                              │  emit_actions   │
                              └─────────────────┘
```

- **`classify_intent`** — single LLM call, returns `scheduling` or
  `off-topic`. Cheapest model is fine.
- **`extract_params`** — LLM with structured output (LangChain
  `with_structured_output(SchedulingIntent)`) producing a typed
  intent: which entity (user / pt), what mutation, what fields.
- **`resolve_user`** — pure Python: case-insensitive lookup against
  `req.state.users`; route to `ask_clarification` if creation requires
  missing fields.
- **`emit_actions`** — builds the final `ChatResponse` with `reply` and
  the list of `ChatAction`s.

Use `langchain-openai`'s `ChatOpenAI(model="gpt-4o-mini")` (or swap to
`langchain-anthropic`'s `ChatAnthropic`). The `OPENAI_API_KEY` env var is
already configured in the repo `.env`.

## 8. Statelessness

The backend stays stateless (mirrors `/api/solve`). Each `POST
/scheduler/api/chat`:

- carries the latest `message`,
- the full prior `history`,
- and a `state` snapshot of current users + PT availability.

The agent **must not** keep memory across requests. Multi-turn
clarification works because the frontend re-sends `history` next turn.

## 9. Environment variables

- `OPENAI_API_KEY` — required if using `langchain-openai` (already in
  `.env`).
- `ANTHROPIC_API_KEY` — alternative if using `langchain-anthropic`.
- `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` — optional, for
  LangSmith traces.

## 10. Open questions / future work

- **Multi-action ordering.** Today actions are applied in array order.
  Should creates run before adds? (Currently yes, by convention.)
- **Undo.** No undo channel from chat. Could be a future
  `undo_last_chat_actions` button stored in `chatHistory`.
- **Streaming.** Reply is returned in one shot. Streaming partial tokens
  would require swapping the route to SSE.
- **Confirmation step for destructive actions** (`clear_availability`,
  delete user). Could set `waiting_for_input=true` plus a pending action
  preview in `reply`.
- **Group slots.** Not in v1 — current action vocabulary covers users +
  availability only.

## 11. References

- Frontend chat panel: [`projects/scheduler/frontend/index.html`](../frontend/index.html),
  [`app.js`](../frontend/app.js) (`sendChatMessage`, `applyChatAction`,
  `renderChat`).
- Backend stub & Pydantic models: [`projects/scheduler/backend/chat.py`](../backend/chat.py).
- Existing data model & validation: [`projects/scheduler/backend/users.py`](../backend/users.py)
  (slot duration rules, `Day` enum, `hhmm_to_minutes`, `minutes_to_hhmm`).
