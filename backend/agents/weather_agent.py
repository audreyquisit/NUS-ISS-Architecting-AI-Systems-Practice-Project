from typing import Optional

from services.weather_service import get_weather


def run(
    task: str,
    parsed_request: Optional[dict] = None,
    candidate_centres: Optional[list[str]] = None,
):

    print("Weather Agent")

    weather = get_weather()

    return {
        "agent": "weather",
        "task": task,
        "weather": weather,
        "candidate_centres": candidate_centres or [],
        "limitations": ["Weather service currently returns mock data."],
    }
