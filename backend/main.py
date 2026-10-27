import os
import hmac
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import monotonic
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pydantic import ValidationError

load_dotenv(Path(__file__).with_name(".env"))

from agents.orchestrator import build_workflow, _canonical_worker_task
from schemas import UserPreferences
from agents.location_agent import LocationResolutionError, run as run_location_agent
from models.location_models import LocationQuery
from services.location_service import LocationServiceError
from services.hawker_service import list_menu_stalls, search_menu_items
from services.trace_store import save_trace
from services.activity_log import log_activity
from services import session_store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
MAX_MESSAGE_CHARS = 4000


def _trace_access_allowed() -> bool:
    expected = os.getenv("TRACE_ACCESS_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if supplied.startswith("Bearer "):
        supplied = supplied[7:]
    return bool(len(expected) >= 32 and supplied and hmac.compare_digest(supplied, expected))


def _now_iso() -> str:
    sgt = timezone(timedelta(hours=8), name="SGT")
    return datetime.now(sgt).isoformat(timespec="microseconds")


def run_orchestrator(
    message: str, payload: Optional[dict] = None, request_id: Optional[str] = None,
):
    payload = payload or {}
    workflow = build_workflow()
    result = workflow.invoke({
        "user_request": message,
        "request_id": request_id,
        "tasks": [],
        "agent_results": [],
        "final_recommendation": "",
        "conversation_context": payload.get("conversation_context", {}),
        "user_location": payload.get("user_location"),
        "user_location_source": payload.get("user_location_source") or (
            "browser_current_location" if payload.get("user_location") else None
        ),
    })
    agent_results = result.get("agent_results", [])
    raw_location = result.get("location_result") or {}
    location_envelope = next(
        (
            item for item in result.get("verified_results", [])
            if item.get("agent") == "location"
        ),
        None,
    )
    location_data = (
        raw_location if result.get("needs_location")
        else (location_envelope or {}).get("payload", raw_location)
    )
    verified_results = result.get("verified_results", [])
    parsed_request = result.get("parsed_request", {})
    worker_plan = [
        {"agent": task.agent, "task": _canonical_worker_task(task.agent, parsed_request)}
        for task in result.get("tasks", [])
    ]
    return {
        "reply": result.get("final_recommendation", ""),
        "structured": {
            "parsed_request": parsed_request,
            "worker_plan": worker_plan,
            "verified_results": verified_results,
        },
        "location_data": location_data,
        "agent_results": agent_results,
        "parsed_request": parsed_request,
        "verified_results": verified_results,
        "needs_location": result.get("needs_location", False),
        "worker_plan": worker_plan,
    }


def _safe_location_result(location_data: dict) -> dict:
    """Keep the reusable shortlist without coordinates or street addresses."""
    selected = []
    for item in (location_data.get("selected_centres") or []):
        if not isinstance(item, dict) or not item.get("centre_id"):
            continue
        selected.append({
            key: item.get(key)
            for key in ("centre_id", "rank", "logistics_fit", "rationale")
            if item.get(key) is not None
        })
    logistics = []
    for item in (location_data.get("logistics") or []):
        if not isinstance(item, dict) or not item.get("centre_id"):
            continue
        travel = item.get("travel") if isinstance(item.get("travel"), dict) else {}
        logistics.append({
            "centre_id": item.get("centre_id"),
            "centre_name": item.get("centre_name"),
            "travel": {
                key: travel[key]
                for key in (
                    "total_duration_min", "mode", "transfers",
                    "walking_distance_display", "walking_time_display",
                    "source", "confidence",
                )
                if key in travel
            },
        })
    return {"selected_centres": selected, "logistics": logistics}


def _session_state_patch(response: dict) -> dict:
    """Derive the persistable session state from an orchestrator response.

    Only reusable, non-identifying fields are kept: the place the user named,
    route preferences, the centre shortlist, and previously recommended menu
    items. Coordinates, street addresses, and credentials are never stored.
    """
    location_data = response.get("location_data")
    location_data = location_data if isinstance(location_data, dict) else {}
    parsed = response.get("parsed_request")
    parsed = parsed if isinstance(parsed, dict) else {}
    verified = response.get("verified_results")
    verified = verified if isinstance(verified, list) else []
    interpretation = location_data.get("intent_interpretation")
    interpretation = interpretation if isinstance(interpretation, dict) else {}

    joined = next((
        item for item in verified
        if isinstance(item, dict) and item.get("agent") == "candidate_join"
    ), {})
    candidates = joined.get("candidates") if isinstance(joined, dict) else []
    previous_ids = [
        str(item.get("menu_item_id"))
        for item in (candidates or [])
        if isinstance(item, dict) and item.get("menu_item_id")
    ]
    weather = next((
        item for item in verified
        if isinstance(item, dict) and item.get("agent") == "weather"
    ), {})
    weather_payload = weather.get("payload") if isinstance(weather, dict) else {}
    weather_data = (
        weather_payload.get("weather") if isinstance(weather_payload, dict) else {}
    )
    condition = weather_data.get("condition") if isinstance(weather_data, dict) else None

    origin_source = str(location_data.get("origin_source") or "")
    state: dict[str, object] = {
        "origin_source": origin_source or None,
        "travel_mode": interpretation.get("travel_mode") or parsed.get("travel_mode"),
        "max_travel_time_min": parsed.get("max_travel_time_min"),
        "priorities": [str(item) for item in (parsed.get("priorities") or [])],
        "previous_menu_item_ids": previous_ids,
        "last_weather": str(condition) if condition else None,
    }
    if origin_source == "free_text_place":
        # Only a place the user typed is safe to carry forward. A
        # reverse-geocoded label would expose the user's own address.
        state["location_text"] = str(location_data.get("origin_label") or "")
    shortlist = _safe_location_result(location_data)
    if shortlist["selected_centres"]:
        if origin_source == "free_text_place":
            shortlist["origin_label"] = str(location_data.get("origin_label") or "")
        state["location_result"] = shortlist
    return {key: value for key, value in state.items() if value not in (None, "", [], {})}


def _resolve_session(payload: dict) -> tuple[Optional[str], dict]:
    """Resolve the session for a request and load its stored context.

    The stored context is canonical once a session holds one. A sessionless
    request keeps using the context supplied by the client.
    """
    try:
        session_id = session_store.ensure_session(payload.get("session_id"))
    except Exception:
        app.logger.exception("Session resolution failed")
        return None, payload
    try:
        stored = session_store.get_context(session_id)
    except Exception:
        app.logger.exception("Session context load failed")
        return session_id, payload
    if not stored.get("turns") and not stored.get("state"):
        return session_id, payload
    return session_id, {**payload, "conversation_context": stored}


def _append_session_turn(
    session_id: Optional[str], role: str, content: str, request_id: str,
) -> None:
    if not session_id:
        return
    try:
        session_store.append_turn(
            session_id, role, content, {"request_id": request_id},
        )
    except Exception:
        app.logger.exception("Session turn append failed")


def _update_session_state(session_id: Optional[str], response: dict) -> None:
    if not session_id:
        return
    try:
        session_store.update_state(session_id, _session_state_patch(response))
    except Exception:
        app.logger.exception("Session state update failed")


def _execute_and_trace(
    message: str, payload: dict, endpoint: str = "chat",
) -> dict:
    from services.activity_log import begin_activity_capture, end_activity_capture, log_activity

    request_id = str(uuid.uuid4())
    session_id, resolved_payload = _resolve_session(payload)
    timestamp_start = _now_iso()
    started = monotonic()
    capture_token, events = begin_activity_capture()
    log_activity(request_id, "request.received", endpoint=endpoint, message_chars=len(message))
    _append_session_turn(session_id, "user", message, request_id)
    try:
        try:
            response = run_orchestrator(message, resolved_payload, request_id=request_id)
        except Exception as exc:
            log_activity(request_id, "request.failed", error_type=type(exc).__name__)
            timestamp_end = _now_iso()
            try:
                save_trace({
                    "request_id": request_id,
                    "timestamp_start": timestamp_start,
                    "timestamp_end": timestamp_end,
                    "timestamp": timestamp_end,
                    "status": "error",
                    "duration_ms": round((monotonic() - started) * 1000),
                    "metadata": {
                        "model": "gpt-4o",
                        "environment": os.getenv("APP_ENV", "development"),
                        "endpoint": endpoint,
                        "workflow_type": "smarthawker_orchestrator",
                    },
                    "user_query": message,
                    "events": events,
                })
            except Exception:
                log_activity(request_id, "trace.save_failed")
            raise
        log_activity(
            request_id,
            "request.completed",
            outcome="location_required" if response.get("needs_location") else "success",
            reply_chars=len(response.get("reply") or ""),
        )
        timestamp_end = _now_iso()
        try:
            save_trace({
                "request_id": request_id,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "timestamp": timestamp_end,
                "status": "location_required" if response.get("needs_location") else "success",
                "duration_ms": round((monotonic() - started) * 1000),
                "metadata": {
                    "model": "gpt-4o",
                    "environment": os.getenv("APP_ENV", "development"),
                    "endpoint": endpoint,
                    "workflow_type": "smarthawker_orchestrator",
                },
                "user_query": message,
                "events": events,
                "response": response,
            })
        except (sqlite3.ProgrammingError, sqlite3.InterfaceError,
                sqlite3.IntegrityError, sqlite3.NotSupportedError):
            raise
        except (OSError, sqlite3.DatabaseError) as error:
            # Optional persistence must not discard a successful response or
            # prevent the independent session updates below. Never log payloads
            # or provider/storage exception messages here.
            log_activity(request_id, "trace.save_failed", error_type=type(error).__name__)
        _append_session_turn(
            session_id, "assistant", response.get("reply") or "", request_id,
        )
        _update_session_state(session_id, response)
        return {"request_id": request_id, "session_id": session_id, **response}
    finally:
        end_activity_capture(capture_token)


def build_preferences_message(payload: dict) -> str:
    preferences = UserPreferences.model_validate(payload)
    dietary = ", ".join(preferences.dietary_preferences) if preferences.dietary_preferences else "no specific dietary restriction"
    weather = preferences.weather or "not specified"
    max_queue = preferences.max_queue if preferences.max_queue is not None else "not specified"

    return (
        f"I want food in {preferences.location}. "
        f"Dietary preferences: {dietary}. "
        f"Budget: {preferences.budget}. "
        f"Available time: {preferences.available_time_min} minutes. "
        f"Weather: {weather}. "
        f"Maximum queue: {max_queue}."
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    raw_message = payload.get("message")
    if raw_message is not None and not isinstance(raw_message, str):
        return jsonify({"error": "Message must be text."}), 400
    message = (raw_message or "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message must be {MAX_MESSAGE_CHARS} characters or fewer."}), 413

    wrapped = _execute_and_trace(message, payload, endpoint="chat")
    if wrapped.get("needs_location"):
        location_error = (wrapped.get("location_data") or {}).get("error")
        return jsonify({
            "code": (
                "location_required"
                if (wrapped.get("location_data") or {}).get("location_required")
                else "location_resolution_failed"
            ),
            "error": location_error or (
                "I need a starting point to find nearby hawker centres. "
                "Allow location access or name a nearby place."
            ),
            "session_id": wrapped.get("session_id"),
        }), 422
    return jsonify({
        "request_id": wrapped["request_id"],
        "session_id": wrapped.get("session_id"),
        "reply": wrapped["reply"],
        "structured": wrapped["structured"],
        "location_data": wrapped.get("location_data"),
        "parsed_request": wrapped.get("parsed_request"),
        "verified_results": wrapped.get("verified_results"),
        "worker_plan": wrapped.get("worker_plan"),
    })


@app.route("/api/recommendation", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True) or {}

    raw_message = payload.get("message")
    if raw_message is not None and not isinstance(raw_message, str):
        return jsonify({"error": "Message must be text."}), 400
    message = (raw_message or "").strip()
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message must be {MAX_MESSAGE_CHARS} characters or fewer."}), 413
    if message:
        wrapped = _execute_and_trace(message, payload, endpoint="recommendation")
        return jsonify({
            "request_id": wrapped["request_id"],
            "session_id": wrapped.get("session_id"),
            "reply": wrapped["reply"],
            "structured": wrapped["structured"],
            "location_data": wrapped.get("location_data"),
            "parsed_request": wrapped.get("parsed_request"),
            "verified_results": wrapped.get("verified_results"),
            "worker_plan": wrapped.get("worker_plan"),
        })

    if any(key in payload for key in ["location", "budget", "available_time_min", "dietary_preferences"]):
        try:
            message = build_preferences_message(payload)
            wrapped = _execute_and_trace(message, payload, endpoint="recommendation")
            return jsonify({
                "request_id": wrapped["request_id"],
                "session_id": wrapped.get("session_id"),
                "reply": wrapped["reply"],
                "structured": wrapped["structured"],
                "location_data": wrapped.get("location_data"),
                "parsed_request": wrapped.get("parsed_request"),
                "verified_results": wrapped.get("verified_results"),
                "worker_plan": wrapped.get("worker_plan"),
            })
        except Exception as exc:
            return jsonify({"error": f"Invalid preferences payload: {str(exc)}"}), 400

    return jsonify({"error": "Message or valid preferences are required."}), 400


@app.route("/api/traces", methods=["GET"])
def api_traces():
    if not _trace_access_allowed():
        return jsonify({"error": "Not found."}), 404
    from services.trace_store import list_traces
    return jsonify(list_traces())


@app.route("/api/trace/<request_id>", methods=["GET"])
def api_trace(request_id: str):
    if not _trace_access_allowed() or not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        request_id,
    ):
        return jsonify({"error": "Not found."}), 404
    from services.trace_store import load_trace
    trace = load_trace(request_id)
    if not trace:
        return jsonify({"error": "Trace not found"}), 404
    return jsonify(trace)


@app.route("/api/heartbeat", methods=["POST"])
def heartbeat():
    payload = request.get_json(silent=True) or {}
    session_state = payload.get("session_state", {})
    alerts = []

    current_weather = payload.get("current_weather")
    previous_weather = session_state.get("last_weather")
    if current_weather and previous_weather and current_weather != previous_weather:
        alerts.append({
            "type": "weather_change",
            "message": f"Weather changed from {previous_weather} to {current_weather}. Consider adjusting plans.",
        })

    return jsonify({
        "status": "ok",
        "alerts": alerts,
        "needs_review": bool(alerts),
    })


@app.route("/api/stalls", methods=["GET"])
def list_stalls():
    try:
        limit = min(max(int(request.args.get("limit", "50")), 1), 200)
        offset = max(int(request.args.get("offset", "0")), 0)
    except ValueError:
        return jsonify({"error": "limit and offset must be whole numbers."}), 400
    stalls = list_menu_stalls(limit=limit, offset=offset)
    return jsonify({
        "source": "Wak-Wak listings plus clearly marked synthetic demo entries; centre IDs matched to NEA data",
        "count": len(stalls),
        "limit": limit,
        "offset": offset,
        "stalls": stalls,
    })


@app.route("/api/menu", methods=["GET"])
def menu_lookup():
    query = (request.args.get("q") or "").strip()
    centre_id = (request.args.get("centre_id") or "").strip() or None
    raw_budget = request.args.get("budget")
    try:
        budget = float(raw_budget) if raw_budget is not None else None
        limit = min(max(int(request.args.get("limit", "20")), 1), 100)
    except ValueError:
        return jsonify({"error": "budget must be numeric and limit must be a whole number."}), 400
    if budget is not None and budget < 0:
        return jsonify({"error": "budget must be zero or greater."}), 400
    if not query and not centre_id:
        return jsonify({"error": "Provide q or centre_id to narrow the menu lookup."}), 400

    items = search_menu_items(query, centre_id, budget, limit)
    return jsonify({
        "source": "Wak-Wak listings plus clearly marked synthetic demo entries",
        "price_currency": "SGD",
        "price_note": "Source prices are not guaranteed current. Entries marked is_mock are illustrative, synthetic prices.",
        "count": len(items),
        "items": items,
    })


@app.route("/api/location", methods=["POST"])
def evaluate_location():
    """Expose the Location & Logistics worker directly for diagnostics."""
    payload = request.get_json(silent=True) or {}
    if "request" not in payload:
        payload["request"] = payload.get("message", payload.get("task", ""))
    if not isinstance(payload.get("request"), str):
        return jsonify({"error": "Request must be text."}), 400
    if len(payload.get("request") or "") > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Request must be {MAX_MESSAGE_CHARS} characters or fewer."}), 413
    try:
        query = LocationQuery.model_validate(payload)
    except ValidationError as error:
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return jsonify({"error": "Invalid location request", "details": details}), 400
    try:
        return jsonify(run_location_agent(query))
    except LocationResolutionError as error:
        return jsonify({"error": str(error)}), 422
    except LocationServiceError as error:
        return jsonify({"error": str(error)}), 503
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Location & Logistics agent failed")
        return jsonify({"error": "Location assessment failed."}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
