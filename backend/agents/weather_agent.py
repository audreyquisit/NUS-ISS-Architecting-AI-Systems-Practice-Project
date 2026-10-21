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

    limitations = list(weather.get("limitations") or [])
    if weather.get("condition") == "unavailable":
        confidence = 0.0
        limitations.append(
            "The NEA weather forecast could not be read, so no forecast is "
            "shown."
        )
    else:
        confidence = 0.8

    return {
        "agent": "weather",
        "confidence": confidence,
        "task": task,
        "weather": weather,
        "candidate_centres": centres,
        "limitations": limitations,
    }
