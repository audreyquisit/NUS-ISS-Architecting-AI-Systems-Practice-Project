from typing import List, Optional


def _crowd_level_from_wait(queue_minutes: int) -> str:
    if queue_minutes <= 5:
        return "Low"
    if queue_minutes <= 15:
        return "Moderate"
    if queue_minutes <= 25:
        return "High"
    return "Very High"


def get_queue_estimates(task: Optional[str] = None, stalls: Optional[List[dict]] = None):
    """Return deterministic queue and crowd estimates for hawker stalls.

    The task string can describe a user preference such as "under 10 minutes".
    When present, we use it to derive a queue tolerance and expose a simple
    crowd-level label for each candidate.
    """

    base_stalls = [
        {
            "stall_name": "Example Vegetarian Stall",
            "hawker_centre": "Tekka Centre",
            "base_queue_minutes": 5,
        },
        {
            "stall_name": "Example Noodle Stall",
            "hawker_centre": "Maxwell Food Centre",
            "base_queue_minutes": 12,
        },
        {
            "stall_name": "Char Kway Teow Corner",
            "hawker_centre": "Maxwell Food Centre",
            "base_queue_minutes": 18,
        },
    ]

    queue_limit = None
    if task:
        task_lower = task.lower()
        if "under 5" in task_lower or "5 minutes" in task_lower:
            queue_limit = 5
        elif "under 10" in task_lower or "10 minutes" in task_lower:
            queue_limit = 10
        elif "under 15" in task_lower or "15 minutes" in task_lower:
            queue_limit = 15
        elif "under 20" in task_lower or "20 minutes" in task_lower:
            queue_limit = 20

    queue_candidates = []

    for stall in (stalls or base_stalls):
        stall_name = stall.get("stall_name", "Unknown Stall")
        hawker_centre = stall.get("hawker_centre", "Unknown Hawker Centre")
        queue_minutes = int(stall.get("base_queue_minutes", stall.get("queue_minutes", 0)))
        crowd_level = _crowd_level_from_wait(queue_minutes)

        candidate = {
            "stall_name": stall_name,
            "hawker_centre": hawker_centre,
            "queue_minutes": queue_minutes,
            "crowd_level": crowd_level,
            "wait_band": "Short" if queue_minutes <= 5 else "Moderate" if queue_minutes <= 15 else "Long",
            "fits_queue_limit": True if queue_limit is None else queue_minutes <= queue_limit,
        }

        queue_candidates.append(candidate)

    return queue_candidates
