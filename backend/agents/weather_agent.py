from services.weather_service import get_weather


def run(task: str):

    print("Weather Agent")

    weather = get_weather()

    return {
        "agent": "weather",
        "task": task,
        "weather": weather
    }
