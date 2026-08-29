import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# TODO: Replace mock response with real backend call once endpoints are ready.
# Expected backend endpoint: POST http://localhost:8000/api/recommendation
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

    # Mock reply so the frontend is usable before backend endpoints exist.
    # Replace the body below with a requests.post(BACKEND_URL + "/api/recommendation", ...)
    mock_reply = (
        "Thanks for your message! The backend AI endpoints are not wired up yet. "
        "Once ready, this will return real hawker stall recommendations based on your preferences."
    )

    return jsonify({"reply": mock_reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
