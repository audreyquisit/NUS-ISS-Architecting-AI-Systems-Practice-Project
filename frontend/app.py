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
            json={"message": user_message},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return jsonify({"reply": data.get("reply", "No recommendation returned.")})
    except Exception as exc:
        return jsonify({"error": f"Backend call failed: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
