from datetime import datetime, timedelta
from typing import List, Optional


def _now_iso() -> str:
    return (datetime.now() + timedelta(minutes=0)).strftime("%Y-%m-%dT%H:%M:%S%z")


REGION_QUEUE_FEED = {
    "central": [
        {
            "stall_name": "Hainanese Chicken Rice",
            "hawker_centre": "Maxwell Food Centre",
            "region": "Central",
            "queue_minutes": 18,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 7,
            "queue_score": 58,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Vegetarian Delight",
            "hawker_centre": "Tekka Centre",
            "region": "Central",
            "queue_minutes": 11,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 5,
            "queue_score": 70,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Fried Kway Teow Corner",
            "hawker_centre": "Tiong Bahru Market",
            "region": "Central",
            "queue_minutes": 15,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 6,
            "queue_score": 64,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
    "east": [
        {
            "stall_name": "Curry Chicken Bee Hoon",
            "hawker_centre": "Bedok 85 Market",
            "region": "East",
            "queue_minutes": 22,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 8,
            "queue_score": 51,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Bak Kut Teh",
            "hawker_centre": "Tampines Round Market",
            "region": "East",
            "queue_minutes": 13,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 6,
            "queue_score": 68,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Sambal Stingray",
            "hawker_centre": "East Coast Lagoon Food Village",
            "region": "East",
            "queue_minutes": 19,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 7,
            "queue_score": 56,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
    "west": [
        {
            "stall_name": "Char Kway Teow",
            "hawker_centre": "Jurong West 505 Food Centre",
            "region": "West",
            "queue_minutes": 25,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 8,
            "queue_score": 46,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Nasi Lemak Corner",
            "hawker_centre": "Clementi 448 Market",
            "region": "West",
            "queue_minutes": 10,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 5,
            "queue_score": 73,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Indian Muslim Mee Goreng",
            "hawker_centre": "Boon Lay Place Food Village",
            "region": "West",
            "queue_minutes": 16,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 6,
            "queue_score": 62,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
    "north": [
        {
            "stall_name": "Nasi Lemak Stall",
            "hawker_centre": "Woodlands Mart",
            "region": "North",
            "queue_minutes": 9,
            "crowd_level": "Low",
            "wait_band": "Short",
            "estimated_service_time_minutes": 4,
            "queue_score": 78,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Yong Tau Foo",
            "hawker_centre": "Yishun Park Hawker Centre",
            "region": "North",
            "queue_minutes": 14,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 6,
            "queue_score": 66,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Prawn Noodle Stall",
            "hawker_centre": "Sembawang Food Centre",
            "region": "North",
            "queue_minutes": 17,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 7,
            "queue_score": 60,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
    "north_east": [
        {
            "stall_name": "Mala Xiang Guo",
            "hawker_centre": "Hougang 1 Food Centre",
            "region": "North-East",
            "queue_minutes": 20,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 8,
            "queue_score": 54,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Indian Rojak",
            "hawker_centre": "Serangoon Garden Market",
            "region": "North-East",
            "queue_minutes": 12,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 5,
            "queue_score": 70,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Teochew Fish Soup",
            "hawker_centre": "Punggol Food Centre",
            "region": "North-East",
            "queue_minutes": 15,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 6,
            "queue_score": 64,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
    "south": [
        {
            "stall_name": "Laksa Stall",
            "hawker_centre": "Queenstown Market",
            "region": "South",
            "queue_minutes": 10,
            "crowd_level": "Moderate",
            "wait_band": "Moderate",
            "estimated_service_time_minutes": 5,
            "queue_score": 74,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
        {
            "stall_name": "Chicken Rice Specialist",
            "hawker_centre": "Dunman Food Centre",
            "region": "South",
            "queue_minutes": 19,
            "crowd_level": "High",
            "wait_band": "Long",
            "estimated_service_time_minutes": 7,
            "queue_score": 57,
            "observed_at": _now_iso(),
            "source": "live_queue_feed",
        },
    ],
}


def get_live_queue_feed(task: Optional[str] = None) -> List[dict]:
    region_aliases = {
        "central": ["central", "raffles", "marina", "city", "bugis", "orchard", "cbd", "tanjong", "clarke quay"],
        "east": ["east", "bedok", "tampines", "marine parade", "eunos", "paya lebar"],
        "west": ["west", "jurong", "clementi", "boon lay", "choa chu kang", "bukit batok"],
        "north": ["north", "woodlands", "yishun", "sembawang", "admiralty", "khatib"],
        "north_east": ["north-east", "northeast", "hougang", "serangoon", "punggol", "sengkang"],
        "south": ["south", "queenstown", "buona vista", "pasir panjang", "dunman", "katong"],
    }

    task_lower = (task or "").lower()
    matched_region = None

    for region_name, aliases in region_aliases.items():
        if any(alias in task_lower for alias in aliases):
            matched_region = region_name
            break

    if matched_region:
        return REGION_QUEUE_FEED.get(matched_region, [])

    all_candidates: List[dict] = []
    for region_candidates in REGION_QUEUE_FEED.values():
        all_candidates.extend(region_candidates)

    return all_candidates
