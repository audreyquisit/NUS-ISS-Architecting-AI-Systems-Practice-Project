import os
import ipaddress
import uuid
from pathlib import Path
from dotenv import load_dotenv
import requests
from flask import Flask, render_template, request, jsonify

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _trace_client_allowed(remote_addr: str) -> bool:
    """Allow loopback clients, and private-network clients when enabled.

    The trace viewer carries operational evidence, so it stays loopback-only
    unless `TRACE_VIEWER_ALLOW_REMOTE` explicitly opens it to local networks.
    """
    try:
        address = ipaddress.ip_address(remote_addr or "")
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if os.getenv("TRACE_VIEWER_ALLOW_REMOTE", "").casefold() != "true":
        return False
    return address.is_private or address.is_link_local


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/trace/<request_id>")
def trace_viewer(request_id):
    if os.getenv("ENABLE_TRACE_VIEWER", "").casefold() != "true":
        return "Not found", 404
    if not _trace_client_allowed(request.remote_addr or ""):
        return "Not found", 404
    try:
        if str(uuid.UUID(request_id)) != request_id.casefold():
            return "Not found", 404
    except (ValueError, AttributeError):
        return "Not found", 404
    token = os.getenv("TRACE_ACCESS_TOKEN", "")
    if not token:
        return "Trace access is not configured", 503
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/trace/{request_id}",
            headers={"Authorization": f"Bearer {token}"}, timeout=30,
        )
        if response.status_code != 200:
            return "Trace not found", 404
        return render_template("trace.html", trace=response.json())
    except requests.RequestException:
        return "Backend call failed", 502


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": user_message,
                "user_location": payload.get("user_location"),
                "conversation_context": payload.get("conversation_context", {}),
                "session_id": payload.get("session_id"),
            },
            timeout=90,
        )
        return (jsonify(response.json()), response.status_code)
    except requests.RequestException as exc:
        return jsonify({"error": f"Backend call failed: {exc}"}), 502
    except ValueError:
        return jsonify({"error": "Backend returned an invalid response."}), 502


@app.route("/api/heartbeat", methods=["POST"])
def heartbeat():
    payload = request.get_json(silent=True) or {}
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/heartbeat",
            json=payload,
            timeout=30,
        )
        return (jsonify(response.json()), response.status_code)
    except requests.RequestException as exc:
        return jsonify({"error": f"Backend call failed: {exc}"}), 502
    except ValueError:
        return jsonify({"error": "Backend returned an invalid response."}), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
