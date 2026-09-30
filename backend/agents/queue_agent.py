from typing import Optional

from services.live_queue_feed import get_live_queue_feed
from services.hawker_service import get_menu_stall_id


def run(
    task: str,
    user_request: Optional[str] = None,
    parsed_request: Optional[dict] = None,
    candidate_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
):

    print("Queue Agent")

    context = parsed_request or {}
    query = " ".join(filter(None, [task, user_request, context.get("requested_area")]))
    queue_data = get_live_queue_feed(task=query)
    if candidate_centres:
        queue_data = [
            item for item in queue_data
            if any(_same_centre(item.get("hawker_centre", ""), name) for name in candidate_centres)
        ]
    for item in queue_data:
        queue_centre = item.get("hawker_centre", "")
        item["hawker_centre_id"] = next((
            centre_id
            for name, centre_id in zip(candidate_centres or [], candidate_centre_ids or [])
            if centre_id and _same_centre(queue_centre, name)
        ), None)
        item["stall_id"] = get_menu_stall_id(
            item.get("stall_name", ""), item["hawker_centre_id"]
        )

    return {
        "agent": "queue",
        "task": task,
        "candidates": queue_data,
        "limitations": (
            ["No queue records matched the Location agent's centre shortlist."]
            if candidate_centres and not queue_data else []
        ),
    }


def _same_centre(first: str, second: str) -> bool:
    normalize = lambda value: " ".join(value.casefold().split())
    left, right = normalize(first), normalize(second)
    return bool(left and right and (left in right or right in left))


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())
