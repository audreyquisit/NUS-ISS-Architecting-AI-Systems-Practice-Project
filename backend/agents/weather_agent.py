from typing import Optional

from services.weather_service import get_weather


def run(
    task: str,
    parsed_request: Optional[dict] = None,
    candidate_centres: Optional[list[str]] = None,
):

    requested_area = (parsed_request or {}).get("requested_area")
    centres = candidate_centres or []
    weather = get_weather(area=requested_area, centres=centres)

    confidence = 0.0
    if isinstance(weather, dict) and weather.get("mock") is not True:
        confidence = 0.8
    elif isinstance(weather, dict) and weather.get("mock") is True:
        confidence = 0.0

    return {
        "agent": "weather",
        "confidence": confidence,
        "task": task,
        "weather": weather,
        "candidate_centres": centres,
        "limitations": ["Weather service currently returns mock data."] if confidence == 0.0 else [],
    }
