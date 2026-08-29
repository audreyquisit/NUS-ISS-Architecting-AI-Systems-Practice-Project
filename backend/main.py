from flask import Flask, jsonify, request

from schemas import StallRecommendation, UserPreferences
from recommendation_service import generate_recommendation

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/recommendation", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True) or {}
    preferences = UserPreferences.model_validate(payload)
    recommendation = generate_recommendation(preferences)
    return jsonify(recommendation.model_dump())


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
