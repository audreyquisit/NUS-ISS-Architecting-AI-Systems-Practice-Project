"""SQLite-backed storage for anonymous sessions, conversation turns, and state.

Sessions are anonymous: there is no user model, so a session is only a UUID
handed back to the client. Every write is redacted with the trace store's
redaction rules so emails, phone numbers, coordinates, addresses, and
credentials never reach the database.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.trace_store import _safe_text, _safe_value

_LOGGER = logging.getLogger("smarthawker.sessions")
_SESSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sessions"))
_DB_PATH = os.path.join(_SESSION_DIR, "sessions.db")
DEFAULT_MAX_TURNS = 20
MAX_TURN_CHARS = 2000
MAX_STATE_CHARS = 20000
MAX_STATE_VALUE_CHARS = 2000
MAX_STATE_LIST_ITEMS = 50
MAX_TURNS_PER_SESSION = 200
_PURGE_INTERVAL_SECONDS = 600
_TURN_ROLES = {"user", "assistant"}
_COORDINATE_KEYS = {
    "lat", "lats", "latitude", "latitudes", "lng", "lon", "long", "longitude",
    "longitudes", "coordinates", "origin", "origin_coordinates",
    "destination_coordinates", "point", "geo", "geometry", "wkt",
}
_ADDRESS_KEYS = {"address", "street_address", "full_address", "postal_code"}
_INITIALIZED_PATHS: set[str] = set()
_LAST_PURGE_AT: dict[str, float] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _valid_session_id(value: Any) -> bool:
    try:
        return str(uuid.UUID(str(value))) == str(value).casefold()
    except (ValueError, AttributeError, TypeError):
        return False


def _retention_days() -> int:
    try:
        return max(1, min(int(os.getenv("SESSION_RETENTION_DAYS", "30")), 365))
    except ValueError:
        return 30


def _connect() -> sqlite3.Connection:
    # Autocommit mode keeps read locks from being held across statements, and
    # write transactions take their lock up front with BEGIN IMMEDIATE. Without
    # that, a read-then-write connection can fail with "database is locked"
    # instead of waiting for the busy timeout.
    connection = sqlite3.connect(_DB_PATH, timeout=10, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _write_transaction(connection: sqlite3.Connection):
    """Run a write transaction that queues behind other writers."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    connection.commit()


def _init_db() -> None:
    """Create the schema once per database file, not once per request."""
    if _DB_PATH in _INITIALIZED_PATHS and os.path.exists(_DB_PATH):
        return
    os.makedirs(_SESSION_DIR, mode=0o700, exist_ok=True)
    try:
        os.chmod(_SESSION_DIR, 0o700)
    except OSError:
        pass
    connection = _connect()
    try:
        # The journal mode is a property of the database file, so it is set
        # once here. Re-running it on every connection would need an exclusive
        # lock and fails with "database is locked" while other writers are busy.
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_turns_session_created
            ON turns (session_id, created_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions (updated_at)
        """)
        connection.commit()
    finally:
        connection.close()
    try:
        os.chmod(_DB_PATH, 0o600)
    except OSError:
        pass
    _INITIALIZED_PATHS.add(_DB_PATH)


def _strip_sensitive(value: Any, key: str = "") -> Any:
    """Remove raw coordinates and street addresses before anything is stored."""
    if key.casefold() in _COORDINATE_KEYS:
        return None
    if key.casefold() in _ADDRESS_KEYS:
        return "[REDACTED_ADDRESS]"
    if isinstance(value, dict):
        return {
            str(item_key): _strip_sensitive(item, str(item_key))
            for item_key, item in value.items()
            if str(item_key).casefold() not in _COORDINATE_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    return value


def _bound_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the stored state small enough to stay cheap and reviewable."""
    bounded: dict[str, Any] = {}
    for key, value in state.items():
        bounded[str(key)] = _bound_value(value)
    while len(json.dumps(bounded, ensure_ascii=False, default=str)) > MAX_STATE_CHARS:
        if not bounded:
            break
        bounded.popitem()
    return bounded


def _bound_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_STATE_VALUE_CHARS]
    if isinstance(value, list):
        return [_bound_value(item) for item in value[:MAX_STATE_LIST_ITEMS]]
    if isinstance(value, dict):
        return {str(key): _bound_value(item) for key, item in value.items()}
    return value


def _safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Redact, bound, and normalize a state patch for storage."""
    if not isinstance(state, dict):
        return {}
    stripped = _strip_sensitive(state)
    if not isinstance(stripped, dict):
        return {}
    safe = _safe_value(stripped)
    return _bound_state(safe if isinstance(safe, dict) else {})


def _is_expired(timestamp: Any) -> bool:
    parsed = _parse_iso(timestamp)
    if parsed is None:
        return True
    cutoff = _now() - timedelta(days=_retention_days())
    return parsed < cutoff


def create_session() -> str:
    """Create a new anonymous session and return its identifier."""
    session_id = str(uuid.uuid4())
    _init_db()
    _purge_if_due()
    now = _iso(_now())
    connection = _connect()
    try:
        with _write_transaction(connection):
            connection.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, state_json) "
                "VALUES (?, ?, ?, '{}')",
                (session_id, now, now),
            )
    finally:
        connection.close()
    return session_id


def ensure_session(session_id: Any = None) -> str:
    """Return a usable session id, creating or resetting the session as needed.

    A missing or malformed identifier yields a fresh session. An unknown but
    well-formed identifier is adopted so a client can keep using it. A session
    older than the retention window is reset instead of being reused.
    """
    candidate = str(session_id or "").strip()
    if not _valid_session_id(candidate):
        return create_session()
    _init_db()
    now = _iso(_now())
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
            (candidate,),
        ).fetchone()
        if row is not None and not _is_expired(row["updated_at"]):
            return candidate
        with _write_transaction(connection):
            connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (candidate,),
            )
            connection.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, state_json) "
                "VALUES (?, ?, ?, '{}')",
                (candidate, now, now),
            )
        return candidate
    finally:
        connection.close()


def session_exists(session_id: Any) -> bool:
    """Report whether a live (non-expired) session is stored."""
    if not _valid_session_id(session_id):
        return False
    _init_db()
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (str(session_id).strip(),),
        ).fetchone()
    finally:
        connection.close()
    return row is not None and not _is_expired(row["updated_at"])


def append_turn(
    session_id: str, role: str, content: str, metadata: Optional[dict] = None,
) -> bool:
    """Append one redacted conversation turn to a session."""
    if not _valid_session_id(session_id):
        return False
    safe_role = str(role or "").strip().casefold()
    if safe_role not in _TURN_ROLES:
        return False
    safe_content = _safe_text(str(content or ""), limit=MAX_TURN_CHARS)
    safe_metadata = None
    if metadata:
        safe_metadata = json.dumps(
            _safe_value(metadata), ensure_ascii=False, default=str,
        )
    now = _iso(_now())
    _init_db()
    connection = _connect()
    try:
        with _write_transaction(connection):
            connection.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, state_json) "
                "VALUES (?, ?, ?, '{}')",
                (str(session_id).strip(), now, now),
            )
            connection.execute(
                "INSERT INTO turns (session_id, role, content, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(session_id).strip(), safe_role, safe_content, now, safe_metadata),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, str(session_id).strip()),
            )
            connection.execute(
                "DELETE FROM turns WHERE session_id = ? AND id NOT IN ("
                "SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (str(session_id).strip(), str(session_id).strip(), MAX_TURNS_PER_SESSION),
            )
    finally:
        connection.close()
    return True


def get_context(session_id: str, max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    """Load the stored conversation context for a session."""
    empty: dict[str, Any] = {"state": {}, "turns": []}
    if not _valid_session_id(session_id):
        return empty
    limit = max(1, min(int(max_turns), MAX_TURNS_PER_SESSION))
    _init_db()
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT state_json, updated_at FROM sessions WHERE session_id = ?",
            (str(session_id).strip(),),
        ).fetchone()
        if row is None or _is_expired(row["updated_at"]):
            return empty
        try:
            state = json.loads(row["state_json"] or "{}")
        except (TypeError, ValueError):
            state = {}
        rows = connection.execute(
            "SELECT role, content FROM turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (str(session_id).strip(), limit),
        ).fetchall()
    finally:
        connection.close()
    turns = [
        {"role": item["role"], "content": item["content"]}
        for item in reversed(rows)
    ]
    return {
        "state": state if isinstance(state, dict) else {},
        "turns": turns,
    }


def update_state(session_id: str, state_patch: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge a state patch into the session state and return the result.

    Only the provided keys are replaced; every other stored key is preserved.
    """
    if not _valid_session_id(session_id):
        return {}
    candidate = str(session_id).strip()
    safe_patch = _safe_state(state_patch)
    now = _iso(_now())
    _init_db()
    connection = _connect()
    try:
        with _write_transaction(connection):
            row = connection.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (candidate,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at, state_json) "
                    "VALUES (?, ?, ?, '{}')",
                    (candidate, now, now),
                )
                current: dict[str, Any] = {}
            else:
                try:
                    stored = json.loads(row["state_json"] or "{}")
                except (TypeError, ValueError):
                    stored = {}
                current = stored if isinstance(stored, dict) else {}
            merged = _bound_state({**current, **safe_patch})
            connection.execute(
                "UPDATE sessions SET state_json = ?, updated_at = ? WHERE session_id = ?",
                (
                    json.dumps(merged, ensure_ascii=False, default=str),
                    now,
                    candidate,
                ),
            )
    finally:
        connection.close()
    return merged


def purge_expired() -> int:
    """Delete sessions older than the retention window and return the count."""
    cutoff = _iso(_now() - timedelta(days=_retention_days()))
    _init_db()
    connection = _connect()
    try:
        with _write_transaction(connection):
            cursor = connection.execute(
                "DELETE FROM sessions WHERE updated_at < ?", (cutoff,),
            )
        return cursor.rowcount or 0
    finally:
        connection.close()


def _purge_if_due() -> None:
    """Sweep expired sessions at most once per interval per database file.

    The sweep is a write, so running it on every request would add write
    contention for no benefit: expired sessions change nothing between sweeps.
    """
    now = time.monotonic()
    last = _LAST_PURGE_AT.get(_DB_PATH, 0.0)
    if now - last < _PURGE_INTERVAL_SECONDS:
        return
    _LAST_PURGE_AT[_DB_PATH] = now
    try:
        purge_expired()
    except Exception:
        _LOGGER.warning("Expired session sweep failed", exc_info=True)
