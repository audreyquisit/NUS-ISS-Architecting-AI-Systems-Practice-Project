# Plan: Persist Trace/Activity Logs

## Context
Existing `backend/services/trace_store.py` stores sanitized request traces as JSON files under `backend/traces/`, with retention, redaction, and dated folders.

## Goal
Provide a durable, queryable storage approach for trace/activity logs without breaking existing file-based behavior.

## Proposed Approach
- Keep file-based traces as source of truth for human-readable audit.
- Add optional SQLite index (`backend/traces/traces.db`) for fast filtering by request_id, intent, timestamp, status.
- `save_trace` writes JSON file then upserts a row into SQLite.
- `list_traces`/`load_trace` read from SQLite when available, falling back to filesystem scan.
- Retention purge deletes both file and DB row.

## Affected Boundaries
- `backend/services/trace_store.py`
- New migration/init helper in same module.

## Risks
- Concurrency across processes; use SQLite WAL and short transactions.
- Schema migration for existing traces; add one-time import.

## Validation
- Unit tests for save/load/list and retention.
- Run existing backend tests.
 