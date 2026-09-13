import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

from agents.orchestrator import build_workflow
from schemas import StallRecommendation, UserPreferences

app = Flask(__name__)


def is_greeting_or_non_food_message(message: str) -> bool:
    cleaned = message.strip().lower()
    if not cleaned:
        return True

    greetings = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "greetings", "hey there", "hi there", "hello there"
    }
    if cleaned in greetings or cleaned.startswith(tuple(greetings)):
        return True

    food_keywords = {
        "food", "hawker", "stall", "eat", "meal", "restaurant", "cuisine",
        "vegetarian", "budget", "queue", "crowd", "hungry", "dinner", "lunch",
        "breakfast", "location", "near", "around", "cheap", "affordable", "rice",
        "noodles", "drink", "recommend"
    }
    return not any(keyword in cleaned for keyword in food_keywords)


def run_orchestrator(message: str):
    if is_greeting_or_non_food_message(message):
        return {
            "reply": (
                "Hi! I can help recommend hawker food based on your location, budget, "
                "dietary needs, and queue preferences. Tell me what you’re craving and where you are."
            ),
            "structured": None,
        }

    workflow = build_workflow()
    result = workflow.invoke({
        "user_request": message,
        "tasks": [],
        "agent_results": [],
        "final_recommendation": "",
        "recommendation_data": {}
    })
    return {
        "reply": result.get("final_recommendation", ""),
        "structured": result.get("recommendation_data") or None,
    }


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

    response = run_orchestrator(message)
    return jsonify({
        "reply": response["reply"],
        "structured": response["structured"],
    })


@app.route("/api/recommendation", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True) or {}

    message = (payload.get("message") or "").strip()
    if message:
        response = run_orchestrator(message)
        return jsonify({
            "reply": response["reply"],
            "structured": response["structured"],
        })

    if any(key in payload for key in ["location", "budget", "available_time_min", "dietary_preferences"]):
        try:
            message = build_preferences_message(payload)
            response = run_orchestrator(message)
            return jsonify({
                "reply": response["reply"],
                "structured": response["structured"],
            })
        except Exception as exc:
            return jsonify({"error": f"Invalid preferences payload: {str(exc)}"}), 400

    return jsonify({"error": "Message or valid preferences are required."}), 400


@app.route("/api/stalls", methods=["GET"])
def list_stalls():
    stalls = [
        StallRecommendation(
            stall_name="Example Noodle Stall",
            hawker_centre="Maxwell Food Centre",
            cuisine="Chinese",
            price_range="Low",
            score=0.92,
            explanation=(
                "Popular choice for budget-friendly noodles and fast "
                "service."
            ),
        ),
        StallRecommendation(
            stall_name="Vegetarian Delight",
            hawker_centre="Tekka Centre",
            cuisine="Vegetarian",
            price_range="Medium",
            score=0.88,
            explanation=(
                "Good option for dietary preferences and low queue "
                "risk."
            ),
        ),
    ]
    return jsonify([stall.model_dump() for stall in stalls])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
