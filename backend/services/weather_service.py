from typing import Optional


_AREAS = {
    "east": "Bedok",
    "north": "Yishun",
    "north_east": "Sengkang",
    "west": "Jurong",
    "south": "Tiong Bahru",
    "central": "Toa Payoh",
}


def get_weather(area: Optional[str] = None, centres: Optional[list[str]] = None):
    location = _AREAS.get(area) if area else None
    if not location and centres:
        location = centres[0]

    return {
        "condition": "rainy",
        "walking_impact": "high",
        "requested_area": area,
        "location": location,
        "centres": centres or [],
        "mock": True,
    }