from typing import List, Optional


def _crowd_level_from_wait(queue_minutes: int) -> str:
    if queue_minutes <= 5:
        return "Low"
    if queue_minutes <= 15:
        return "Moderate"
    if queue_minutes <= 25:
        return "High"
    return "Very High"


def _parse_time_context(task: Optional[str]) -> dict:
    if not task:
        return {
            "time_period": "general",
            "is_weekend": False,
            "is_rainy": False,
            "queue_limit": None,
        }

    task_lower = task.lower()

    if any(token in task_lower for token in ["lunch", "midday", "12pm", "noon", "peak lunch"]):
        time_period = "lunch"
    elif any(token in task_lower for token in ["dinner", "evening", "6pm", "7pm", "8pm", "after work"]):
        time_period = "dinner"
    elif any(token in task_lower for token in ["breakfast", "morning", "9am", "10am", "11am"]):
        time_period = "breakfast"
    else:
        time_period = "general"

    is_weekend = any(token in task_lower for token in ["weekend", "saturday", "sunday"])
    is_rainy = any(token in task_lower for token in ["rain", "rainy", "wet", "storm", "downpour"])

    queue_limit = None
    if "under 5" in task_lower or "5 minutes" in task_lower:
        queue_limit = 5
    elif "under 10" in task_lower or "10 minutes" in task_lower:
        queue_limit = 10
    elif "under 15" in task_lower or "15 minutes" in task_lower:
        queue_limit = 15
    elif "under 20" in task_lower or "20 minutes" in task_lower:
        queue_limit = 20
    elif "under 30" in task_lower or "30 minutes" in task_lower:
        queue_limit = 30

    return {
        "time_period": time_period,
        "is_weekend": is_weekend,
        "is_rainy": is_rainy,
        "queue_limit": queue_limit,
    }


def _time_multiplier(time_period: str) -> float:
    time_map = {
        "breakfast": 0.8,
        "lunch": 1.45,
        "dinner": 1.7,
        "general": 1.1,
    }
    return time_map.get(time_period, 1.1)


def _weather_multiplier(is_rainy: bool) -> float:
    return 1.25 if is_rainy else 1.0


def _day_multiplier(is_weekend: bool) -> float:
    return 1.2 if is_weekend else 1.0


def get_queue_estimates(task: Optional[str] = None, stalls: Optional[List[dict]] = None):
    """Estimate queue times and crowd levels for hawker stalls.

    This is an in-memory, proposal-aligned approximation without persistence.
    It adjusts wait times by typical meal-period patterns, weekend demand,
    and weather, while still remaining deterministic for demo use.
    """

    context = _parse_time_context(task)
    time_period = context["time_period"]
    is_weekend = context["is_weekend"]
    is_rainy = context["is_rainy"]
    queue_limit = context["queue_limit"]

    base_stalls = [
        {
            "stall_name": "Example Vegetarian Stall",
            "hawker_centre": "Tekka Centre",
            "base_queue_minutes": 5,
            "popularity": 1.0,
            "avg_serving_time": 4,
        },
        {
            "stall_name": "Example Noodle Stall",
            "hawker_centre": "Maxwell Food Centre",
            "base_queue_minutes": 12,
            "popularity": 1.25,
            "avg_serving_time": 6,
        },
        {
            "stall_name": "Char Kway Teow Corner",
            "hawker_centre": "Maxwell Food Centre",
            "base_queue_minutes": 18,
            "popularity": 1.6,
            "avg_serving_time": 8,
        },
        {
            "stall_name": "Hainanese Chicken Rice",
            "hawker_centre": "Amoy Street Food Centre",
            "base_queue_minutes": 22,
            "popularity": 1.7,
            "avg_serving_time": 9,
        },
    ]

    queue_candidates = []

    for stall in (stalls or base_stalls):
        stall_name = stall.get("stall_name", "Unknown Stall")
        hawker_centre = stall.get("hawker_centre", "Unknown Hawker Centre")
        base_queue = int(stall.get("base_queue_minutes", stall.get("queue_minutes", 0)))
        popularity = float(stall.get("popularity", 1.0))
        avg_serving_time = int(stall.get("avg_serving_time", 5))

        adjusted_wait = round(
            base_queue
            * popularity
            * _time_multiplier(time_period)
            * _day_multiplier(is_weekend)
            * _weather_multiplier(is_rainy)
        )

        queue_minutes = max(1, adjusted_wait)
        crowd_level = _crowd_level_from_wait(queue_minutes)
        wait_band = (
            "Short" if queue_minutes <= 5
            else "Moderate" if queue_minutes <= 15
            else "Long"
        )
        queue_score = max(0, min(100, round(100 - (queue_minutes * 3) - (avg_serving_time * 2))))
        fits_queue_limit = queue_limit is None or queue_minutes <= queue_limit

        candidate = {
            "stall_name": stall_name,
            "hawker_centre": hawker_centre,
            "queue_minutes": queue_minutes,
            "crowd_level": crowd_level,
            "wait_band": wait_band,
            "estimated_service_time_minutes": avg_serving_time,
            "queue_score": queue_score,
            "fits_queue_limit": fits_queue_limit,
            "crowd_reason": (
                f"Typical crowd during {time_period or 'general'} periods is {crowd_level.lower()} "
                f"with an expected wait of around {queue_minutes} minutes."
            ),
        }

        queue_candidates.append(candidate)

    queue_candidates.sort(key=lambda item: item["queue_minutes"])
    return queue_candidates
