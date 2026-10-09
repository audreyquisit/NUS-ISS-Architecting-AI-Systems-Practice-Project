import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

_TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def _sanitize(data: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(data)
    sanitized.pop("user_request", None)
    sanitized.pop("payload", None)
    response = sanitized.get("response", {})
    if isinstance(response, dict):
        response.pop("location_data", None)
        sanitized["response"] = response
    return sanitized


def save_trace(trace_data: dict[str, Any]) -> str:
    os.makedirs(_TRACE_DIR, exist_ok=True)
    request_id = trace_data.get("request_id") or str(uuid.uuid4())
    path = os.path.join(_TRACE_DIR, f"{request_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(trace_data), f, ensure_ascii=False, indent=2)
    return request_id


def load_trace(request_id: str) -> Optional[dict[str, Any]]:
    path = os.path.join(_TRACE_DIR, f"{request_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    if not os.path.exists(_TRACE_DIR):
        return []
    files = sorted(os.listdir(_TRACE_DIR), reverse=True)[:limit]
    traces = []
    for name in files:
        request_id = name[:-5] if name.endswith(".json") else name
        path = os.path.join(_TRACE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            traces.append({
                "request_id": request_id,
                "timestamp": data.get("timestamp"),
                "intent": (data.get("response", {}).get("parsed_request") or {}).get("intent", ""),
            })
        except Exception:
            continue
    return traces
