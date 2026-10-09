import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pydantic import ValidationError

load_dotenv(Path(__file__).with_name(".env"))

from agents.orchestrator import build_workflow
from schemas import UserPreferences
from agents.location_agent import LocationResolutionError, run as run_location_agent
from models.location_models import LocationQuery
from services.location_service import LocationServiceError
from services.hawker_service import list_menu_stalls, search_menu_items
from services.trace_store import save_trace

app = Flask(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_orchestrator(message: str, payload: Optional[dict] = None):
    payload = payload or {}
    workflow = build_workflow()
    result = workflow.invoke({
        "user_request": message,
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
    worker_plan = [
        task.model_dump() if hasattr(task, "model_dump") else task
        for task in result.get("tasks", [])
    ]
    verified_results = result.get("verified_results", [])
    parsed_request = result.get("parsed_request", {})
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


def _execute_and_trace(message: str, payload: dict) -> dict:
    request_id = str(uuid.uuid4())
    response = run_orchestrator(message, payload)
    save_trace({
        "request_id": request_id,
        "timestamp": _now_iso(),
        "user_request": message,
        "payload": payload,
        "response": response,
    })
    return {"request_id": request_id, **response}


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
    message = (payload.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    wrapped = _execute_and_trace(message, payload)
    if wrapped.get("needs_location"):
        location_error = (wrapped.get("location_data") or {}).get("error")
        return jsonify({"error": location_error or "I need a starting location."}), 422
    return jsonify({
        "request_id": wrapped["request_id"],
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

    message = (payload.get("message") or "").strip()
    if message:
        wrapped = _execute_and_trace(message, payload)
        return jsonify({
            "request_id": wrapped["request_id"],
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
            wrapped = _execute_and_trace(message, payload)
            return jsonify({
                "request_id": wrapped["request_id"],
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
    from services.trace_store import list_traces
    return jsonify(list_traces())


@app.route("/api/trace/<request_id>", methods=["GET"])
def api_trace(request_id: str):
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
