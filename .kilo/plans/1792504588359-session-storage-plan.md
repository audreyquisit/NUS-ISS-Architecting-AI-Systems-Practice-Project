# Plan: Persistent Session/Context/Conversation Storage

## Summary
Add SQLite-backed storage for anonymous sessions, conversation turns, and bounded session state. Follow the existing `trace_store.py` pattern. Server-stored context is canonical when `session_id` is present.

## Decisions
- Anonymous sessions only; no user accounts.
- Client sends `session_id`; backend creates one if missing and returns it.
- Server-stored conversation context is canonical when `session_id` is present.
- Retention: 30 days, matching trace retention.
- SQLite only; no new dependencies.

## Schema
- `sessions(session_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, state_json TEXT)`
- `turns(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at TEXT, metadata_json TEXT, FOREIGN KEY(session_id) REFERENCES sessions(session_id))`
- Index: `turns(session_id, created_at)`.

## Service
Create `backend/services/session_store.py`:
- `create_session() -> str`
- `append_turn(session_id, role, content, metadata=None)`
- `get_context(session_id, max_turns=20) -> {"state": ..., "turns": [...]}`
- `update_state(session_id, state_patch)` shallow-merges keys, replacing only provided keys and preserving prior keys.
- `purge_expired()`

## Integration
- Accept optional `session_id` in `/api/chat` and `/api/recommendation` payloads.
- Resolve/create session before orchestrator.
- Load stored context into `conversation_context`.
- Append user turn before execution and assistant turn after success.
- Return `session_id` in responses.
- Keep `_bounded_planner_context` behavior.

## Security
- Reuse redaction from `trace_store.py`.
- Bound turn content and state sizes.
- Never store secrets, emails, phone numbers, coordinates, or addresses.

## Migration
- Create tables on first use.
- Backward compatible: sessionless requests still accept client context.

## Data Flow
1. Read `session_id` from payload.
2. If missing/invalid, create a new UUID session.
3. Load stored turns/state from SQLite.
4. If stored context exists, set `conversation_context` from storage; otherwise use client context.
5. Append user turn before orchestrator execution.
6. Run orchestrator with resolved context.
7. Append assistant turn and update state after success.
8. Return `session_id` in response.
- Invalid `session_id`: create new session.
- SQLite write failure: log and continue request without persistence.
- Expired session: create new session.

## Validation
- Unit tests for create, append, load, redaction, expiry.
- Integration tests for `/api/chat` session reuse.
- Run backend tests.
