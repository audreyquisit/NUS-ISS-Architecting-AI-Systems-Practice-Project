import os
from typing import Any, Dict, List, Optional

import requests

from services.live_queue_feed import get_live_queue_feed as get_mock_queue_feed

GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACES_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.rating,"
    "places.userRatingCount,places.businessStatus"
)


def _normalise_queue_value(value: Any, default: int = 10) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0:
        return default
    return max(1, int(round(numeric)))


def _crowd_level(queue_minutes: int) -> str:
    if queue_minutes <= 5:
        return "Low"
    if queue_minutes <= 15:
        return "Moderate"
    if queue_minutes <= 25:
        return "High"
    return "Very High"


def _estimate_queue_minutes(result: Dict[str, Any]) -> Optional[int]:
    # Missing or closed operational evidence is not a zero-minute wait.
    if result.get("businessStatus") != "OPERATIONAL":
        return None
    rating = float(result.get("rating") or 0.0)
    total_ratings = int(result.get("userRatingCount") or 0)
    popularity_score = min(100.0, max(0.0, (rating * 18.0) + min(total_ratings, 500) / 10.0))
    queue_proxy = max(2, int(round(100 - popularity_score) / 4))
    return max(3, min(60, queue_proxy))


def _map_place_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    display_name = result.get("displayName") or {}
    place_name = str(display_name.get("text") or "").strip()
    queue_minutes = _estimate_queue_minutes(result)
    if queue_minutes is None:
        return None
    queue_score = max(0, min(100, 100 - (queue_minutes * 3)))
    return {
        "stall_id": result.get("id") or f"google-place-{hash(place_name) % 1000000}",
        "stall_name": place_name,
        "hawker_centre": place_name,
        "hawker_centre_id": None,
        "region": "google_places",
        "base_queue_minutes": queue_minutes,
        "queue_minutes": queue_minutes,
        "crowd_level": _crowd_level(queue_minutes),
        "wait_band": (
            "Short" if queue_minutes <= 5
            else "Moderate" if queue_minutes <= 15
            else "Long"
        ),
        "estimated_service_time_minutes": 5,
        "queue_score": queue_score,
        "source": "google_places",
        "queue_is_mock": False,
        "queue_mock_notice": "Google Places is used as a live place signal; queue minutes are an inferred proxy, not observed wait data.",
        "queue_estimate_type": "popularity_proxy",
        "record_type": "place",
        "business_status": result["businessStatus"],
        "is_mock": False,
        "data_origin": "google_places",
        "place_id": result.get("id"),
        "rating": result.get("rating"),
        "user_ratings_total": result.get("userRatingCount"),
        "formatted_address": result.get("formattedAddress"),
    }


def get_live_queue_feed(task: Optional[str] = None) -> List[dict]:
    """Query Google Places and map place results into the standard queue record schema.

    Note: Google Places does not expose true queue times. This adapter estimates a
    queue proxy from popularity and operational status so the Queue agent keeps
    working with a live feed without changing its interface.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return get_mock_queue_feed(task)

    query = (task or "hawker food centre singapore").strip() or "hawker food centre singapore"
    try:
        response = requests.post(
            GOOGLE_PLACES_BASE_URL,
            headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": GOOGLE_PLACES_FIELD_MASK,
            },
            json={"textQuery": query},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json() or {}
    except Exception:
        return get_mock_queue_feed(task)

    results = payload.get("places") or []
    if not results:
        return get_mock_queue_feed(task)

    queue_records = []
    for result in results[:10]:
        display_name = result.get("displayName") or {}
        place_name = str(display_name.get("text") or "").strip()
        if not place_name:
            continue
        record = _map_place_result(result)
        if record is not None:
            queue_records.append(record)
    return queue_records
