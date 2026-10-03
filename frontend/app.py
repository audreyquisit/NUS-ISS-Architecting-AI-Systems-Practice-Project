import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


@app.route("/")
def home():
    return render_template("index.html")


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
            },
            timeout=90,
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
