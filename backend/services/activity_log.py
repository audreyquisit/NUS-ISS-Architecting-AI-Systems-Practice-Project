"""Compact, request-correlated activity logs for the backend terminal."""

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Optional


_LOGGER = logging.getLogger("smarthawker.activity")
_CAPTURED_EVENTS: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar(
    "smarthawker_captured_activity_events", default=None
)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def log_activity(request_id: Optional[str], event: str, **details: Any) -> None:
    """Write one concise JSON event; avoid raw prompts, credentials and GPS."""
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request_id or "untracked",
        "event": event,
    }
    record.update({key: value for key, value in details.items() if value is not None})
    captured = _CAPTURED_EVENTS.get()
    if captured is not None:
        captured.append(record)
    _LOGGER.info(json.dumps(record, ensure_ascii=False, default=str))


def begin_activity_capture() -> tuple[Token, list[dict[str, Any]]]:
    """Capture this request's activity events for its sanitized trace."""
    events: list[dict[str, Any]] = []
    return _CAPTURED_EVENTS.set(events), events


def end_activity_capture(token: Token) -> None:
    _CAPTURED_EVENTS.reset(token)
