import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

_TRACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "traces"))
_SGT = timezone(timedelta(hours=8), name="SGT")
_TRACE_NAME = re.compile(r"^(\d{6})_(\d{6})__([0-9a-f-]{36})\.json$", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?65[ -]?)?[689]\d{3}[ -]?\d{4}(?!\w)")
_COORD_PAIR = re.compile(r"(?<!\d)-?\d{1,2}\.\d{4,},\s*-?\d{2,3}\.\d{4,}(?!\d)")
_STREET_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[\w .'-]{1,60}\b(?:road|rd|street|st|avenue|ave|drive|dr|lane|ln|"
    r"way|close|crescent|cres|terrace|walk)\b",
    re.I,
)
_API_CREDENTIAL = re.compile(r"\b(?:sk|api|key)-[A-Za-z0-9_-]{16,}\b", re.I)
_SECRET_KEY = re.compile(r"token|password|authorization|api.?key|secret|credential", re.I)


def _safe_text(value: str, limit: int = 2000) -> str:
    text = _EMAIL.sub("[REDACTED_EMAIL]", value)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _COORD_PAIR.sub("[REDACTED_COORDINATES]", text)
    text = _STREET_ADDRESS.sub("[REDACTED_ADDRESS]", text)
    text = _API_CREDENTIAL.sub("[REDACTED_CREDENTIAL]", text)
    return text[:limit] + ("…[truncated]" if len(text) > limit else "")


def _safe_value(value: Any, key: str = "") -> Any:
    key_lower = key.casefold()
    if _SECRET_KEY.search(key_lower):
        return "[REDACTED]"
    if key_lower in {"lat", "latitude", "lng", "lon", "longitude"}:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return "[REDACTED_COORDINATE]"
    if key_lower in {"location", "coordinate", "coordinates", "user_location"}:
        if isinstance(value, (dict, list, tuple)):
            return _safe_value(value)
        return "[REDACTED_LOCATION]"
    if isinstance(value, dict):
        return {str(item_key): _safe_value(item, str(item_key)) for item_key, item in value.items()
                if str(item_key).casefold() not in {"thought", "chain_of_thought", "scratchpad"}}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    return value


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    safe = _safe_value(event)
    if isinstance(safe.get("time"), str):
        safe["time"] = _format_timestamp(_as_sgt(safe["time"]))
    event_name = str(safe.get("event") or "")
    if event_name.startswith("onemap.http.request.started"):
        safe["tool_call"] = {
            "name": f"{safe.get('host', 'OneMap')}{safe.get('endpoint', '')}",
            "arguments": safe.get("params") or safe.get("body_fields") or {},
        }
    elif event_name == "location.onemap_routes.started":
        safe["tool_call"] = {
            "name": "OneMap Routing",
            "arguments": {
                "origin_coordinates": safe.get("origin_coordinates"),
                "route_candidates": safe.get("route_candidates"),
                "requested_mode": safe.get("requested_mode"),
            },
        }
    return safe


def _add_step_latencies(steps: list[dict[str, Any]]) -> None:
    starts: dict[tuple[str, str], datetime] = {}
    for step in steps:
        event_name = str(step.get("event") or "")
        event_time = step.get("time")
        if not isinstance(event_time, str):
            continue
        try:
            timestamp = datetime.fromisoformat(event_time)
        except ValueError:
            continue
        agent = str(step.get("agent") or "")
        if event_name.endswith(".started"):
            starts[(event_name[:-len("started")] + "completed", agent)] = timestamp
        elif event_name.endswith(".completed"):
            started_at = starts.get((event_name, agent))
            if started_at is not None:
                step["latency_ms"] = max(
                    0, round((timestamp - started_at).total_seconds() * 1000)
                )


def _sanitize(data: dict[str, Any]) -> dict[str, Any]:
    """Persist useful operational evidence while redacting sensitive details."""
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    parsed = response.get("parsed_request") or {}
    results = response.get("verified_results") or []
    events = data.get("events")
    if not isinstance(events, list):
        events = data.get("steps") if isinstance(data.get("steps"), list) else []
    steps = [_safe_event(event) for event in events if isinstance(event, dict)]

    # Queue tools return a structured tool trace in the worker evidence rather
    # than emitting activity-log events. Preserve those calls and arguments too.
    for agent_result in response.get("agent_results", []):
        if not isinstance(agent_result, dict) or agent_result.get("agent") != "queue":
            continue
        for evidence in agent_result.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            for order, call in enumerate(evidence.get("tool_trace", []), start=1):
                if isinstance(call, dict):
                    steps.append({
                        "event": "queue.tool_call",
                        "step_order": order,
                        "tool_call": _safe_value(call),
                    })
    _add_step_latencies(steps)

    timestamp_start = _format_timestamp(_as_sgt(
        data.get("timestamp_start") or data.get("timestamp")
    ))
    timestamp_end = _format_timestamp(_as_sgt(
        data.get("timestamp_end") or data.get("timestamp")
    ))
    input_data = data.get("input") if isinstance(data.get("input"), dict) else {}
    user_query = data.get("user_query") or input_data.get("user_query")
    if not isinstance(user_query, str):
        user_query = ""
    output_data = data.get("output") if isinstance(data.get("output"), dict) else {}
    final_response = (
        response.get("reply")
        or output_data.get("final_response")
        or data.get("final_response")
        or ""
    )
    safe_final_response = _safe_text(str(final_response))
    parsed_request = {
        "intent": parsed.get("intent"),
        "search_scope_mode": parsed.get("search_scope_mode"),
        "requested_area": parsed.get("requested_area"),
        "origin_text": _safe_text(str(parsed.get("origin_text")))
        if parsed.get("origin_text") else None,
        "budget_amount": parsed.get("budget_amount"),
        "menu_item_preferences": _safe_value(parsed.get("menu_item_preferences", [])),
        "dietary_preferences": _safe_value(parsed.get("dietary_preferences", [])),
    }
    worker_plan = _safe_value(response.get("worker_plan", []))
    safe_results = [
        {
            "agent": item.get("agent"),
            "status": item.get("status"),
            "verified": item.get("verified"),
            "candidate_count": len(item.get("candidates", []))
            if isinstance(item.get("candidates"), list) else 0,
            "verification": item.get("verification"),
        }
        for item in results if isinstance(item, dict)
    ]
    return {
        "request_id": data.get("request_id"),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "timestamp": timestamp_end,
        "timezone": "Asia/Singapore",
        "status": data.get("status", "success"),
        "duration_ms": data.get("duration_ms"),
        "metadata": _safe_value(data.get("metadata", {})),
        "intent": parsed_request.get("intent"),
        "input": {"user_query": _safe_text(user_query)},
        "steps": steps,
        "response": {
            "parsed_request": parsed_request,
            "worker_plan": worker_plan,
            "verified_results": safe_results,
        },
        "output": {"final_response": safe_final_response},
    }


def _valid_request_id(request_id: str) -> bool:
    try:
        return str(uuid.UUID(request_id)) == request_id.casefold()
    except (ValueError, AttributeError):
        return False


def _as_sgt(value: Any, fallback_timestamp: Optional[float] = None) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(_SGT)
        except ValueError:
            pass
    instant = fallback_timestamp if fallback_timestamp is not None else datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(instant, tz=timezone.utc).astimezone(_SGT)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(_SGT).isoformat(timespec="microseconds")


def _trace_filename(timestamp: datetime, request_id: str) -> str:
    return f"{timestamp:%H%M%S}_{timestamp.microsecond:06d}__{request_id}.json"


def _request_id_from_filename(name: str) -> Optional[str]:
    if name.endswith(".json"):
        legacy_id = name[:-5]
        if _valid_request_id(legacy_id):
            return legacy_id
    match = _TRACE_NAME.fullmatch(name)
    if match and _valid_request_id(match.group(3)):
        return match.group(3)
    return None


def _trace_files():
    if not os.path.isdir(_TRACE_DIR):
        return
    for root, _dirs, names in os.walk(_TRACE_DIR):
        for name in names:
            if name.endswith(".json") and _request_id_from_filename(name):
                yield os.path.join(root, name), name


def _find_trace_path(request_id: str) -> Optional[str]:
    if not os.path.isdir(_TRACE_DIR):
        return None
    for path, name in _trace_files():
        if _request_id_from_filename(name) == request_id:
            return path
    return None


def _update_latest_pointer(path: str) -> None:
    relative = os.path.relpath(path, _TRACE_DIR)
    pointer = os.path.join(_TRACE_DIR, "LATEST.txt")
    temporary = os.path.join(_TRACE_DIR, ".LATEST.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as destination:
            destination.write(relative + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, pointer)
        os.chmod(pointer, 0o600)
    except OSError:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _purge_expired() -> None:
    try:
        retention_days = max(1, min(int(os.getenv("TRACE_RETENTION_DAYS", "30")), 365))
    except ValueError:
        retention_days = 30
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).timestamp()
    if not os.path.isdir(_TRACE_DIR):
        return
    for path, _name in list(_trace_files()):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                continue
            with open(path, "r", encoding="utf-8") as source:
                existing = json.load(source)
            safe = _sanitize(existing)
            if safe != existing:
                with open(path, "w", encoding="utf-8") as destination:
                    json.dump(safe, destination, ensure_ascii=False, indent=2)
                os.chmod(path, 0o600)
        except OSError:
            continue
        except (ValueError, TypeError):
            continue
    for root, dirs, _files in os.walk(_TRACE_DIR, topdown=False):
        for name in dirs:
            directory = os.path.join(root, name)
            try:
                os.rmdir(directory)
            except OSError:
                pass


def _organize_legacy_traces() -> None:
    """Move flat UUID-named traces into dated SGT folders, preserving their IDs."""
    if not os.path.isdir(_TRACE_DIR):
        return
    for name in os.listdir(_TRACE_DIR):
        request_id = _request_id_from_filename(name)
        if not request_id or name != f"{request_id}.json":
            continue
        old_path = os.path.join(_TRACE_DIR, name)
        try:
            with open(old_path, "r", encoding="utf-8") as source:
                trace = json.load(source)
            timestamp = _as_sgt(trace.get("timestamp"), os.path.getmtime(old_path))
            trace["timestamp"] = _format_timestamp(timestamp)
            trace["request_id"] = request_id
            safe = _sanitize(trace)
            dated_dir = os.path.join(_TRACE_DIR, f"{timestamp:%Y%m%d}")
            os.makedirs(dated_dir, mode=0o700, exist_ok=True)
            new_path = os.path.join(dated_dir, _trace_filename(timestamp, request_id))
            with open(new_path, "w", encoding="utf-8") as destination:
                json.dump(safe, destination, ensure_ascii=False, indent=2)
            os.chmod(new_path, 0o600)
            old_stat = os.stat(old_path)
            os.utime(new_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            os.remove(old_path)
        except (OSError, ValueError, TypeError):
            continue


def save_trace(trace_data: dict[str, Any]) -> str:
    os.makedirs(_TRACE_DIR, mode=0o700, exist_ok=True)
    try:
        os.chmod(_TRACE_DIR, 0o700)
    except OSError:
        pass
    _organize_legacy_traces()
    _purge_expired()
    request_id = str(trace_data.get("request_id") or uuid.uuid4())
    if not _valid_request_id(request_id):
        request_id = str(uuid.uuid4())
    timestamp = _as_sgt(trace_data.get("timestamp"))
    stored_data = dict(trace_data)
    stored_data["request_id"] = request_id
    stored_data["timestamp"] = _format_timestamp(timestamp)
    dated_dir = os.path.join(_TRACE_DIR, f"{timestamp:%Y%m%d}")
    os.makedirs(dated_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(dated_dir, 0o700)
    except OSError:
        pass
    path = os.path.join(dated_dir, _trace_filename(timestamp, request_id))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(stored_data), f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _update_latest_pointer(path)
    return request_id


def load_trace(request_id: str) -> Optional[dict[str, Any]]:
    if not _valid_request_id(request_id):
        return None
    _organize_legacy_traces()
    _purge_expired()
    path = _find_trace_path(request_id)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    _organize_legacy_traces()
    _purge_expired()
    if not os.path.exists(_TRACE_DIR):
        return []
    traces = []
    for path, name in _trace_files():
        request_id = _request_id_from_filename(name)
        if not request_id:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            timestamp = _as_sgt(data.get("timestamp"), os.path.getmtime(path))
            traces.append({
                "request_id": request_id,
                "timestamp": _format_timestamp(timestamp),
                "timezone": "Asia/Singapore",
                "intent": (data.get("response", {}).get("parsed_request") or {}).get("intent", ""),
                "file": os.path.relpath(path, _TRACE_DIR),
            })
        except Exception:
            continue
    traces.sort(key=lambda item: item["timestamp"], reverse=True)
    selected = traces[:max(1, min(int(limit), 100))]
    if selected:
        _update_latest_pointer(os.path.join(_TRACE_DIR, selected[0]["file"]))
        selected[0]["latest"] = True
    for item in selected[1:]:
        item["latest"] = False
    return selected
