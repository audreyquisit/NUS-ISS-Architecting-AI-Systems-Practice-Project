"""Read the public NEA/MSS 24-hour weather forecast for Singapore.

The forecast is published on data.gov.sg and needs no key. The service
records where the forecast came from and when it was read, so a reply can
say which forecast it is talking about. When the forecast cannot be read the
service says so and never falls back on a made-up condition.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
HTTP_TIMEOUT_SECONDS = float(os.getenv("WEATHER_TIMEOUT_SECONDS", "12"))
CACHE_SECONDS = float(os.getenv("WEATHER_CACHE_SECONDS", "900"))
SOURCE_NAME = "NEA/MSS 24-hour weather forecast"
SOURCE_KEY = "nea_mss_weather"

SINGAPORE = timezone(timedelta(hours=8))

# The forecast regions of the NEA forecast, with a place in each region that
# the chatbot can name.
_REGION_PLACES = {
    "east": "Bedok",
    "north": "Yishun",
    "west": "Jurong",
    "south": "Tiong Bahru",
    "central": "Toa Payoh",
}

# The regions the request may name, mapped to the forecast region they mean.
_AREAS = {
    "east": "east",
    "north": "north",
    "north_east": "north",
    "west": "west",
    "south": "south",
    "central": "central",
}

# Approximate geographic anchors, used only to place a centre in a forecast
# region when the request does not name one.
_REGION_ANCHORS = {
    "west": (1.335, 103.705),
    "east": (1.350, 103.955),
    "north": (1.435, 103.790),
    "north_east": (1.370, 103.895),
    "south": (1.265, 103.820),
    "central": (1.300, 103.840),
}

_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}


def _fetch() -> Optional[dict[str, Any]]:
    """Return the current forecast record, or None when it cannot be read."""
    now = time.monotonic()
    if _CACHE["payload"] is not None and now - _CACHE["at"] < CACHE_SECONDS:
        return _CACHE["payload"]
    try:
        response = requests.get(FORECAST_URL, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError):
        return None
    records = ((body or {}).get("data") or {}).get("records") or []
    if not records:
        return None
    record = records[0]
    _CACHE["payload"] = record
    _CACHE["at"] = now
    return record


def _walking_impact(condition: str) -> str:
    """Return how much the condition puts a walk to the centre."""
    lowered = str(condition or "").casefold()
    if "thunder" in lowered or "heavy" in lowered or "storm" in lowered:
        return "high"
    if "rain" in lowered or "shower" in lowered:
        return "moderate"
    return "low"


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lng1 = first
    lat2, lng2 = second
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _region_for_centres(centres: list[str]) -> Optional[str]:
    """Return the forecast region that holds the given centres."""
    if not centres:
        return None
    from services.location_service import load_hawker_centres

    wanted = {str(name).strip().casefold() for name in centres if name}
    if not wanted:
        return None
    best: Optional[tuple[float, str]] = None
    for centre in load_hawker_centres():
        names = {
            str(centre.get("name") or "").strip().casefold(),
            str(centre.get("official_name") or "").strip().casefold(),
        }
        if not names & wanted:
            continue
        latitude = centre.get("latitude")
        longitude = centre.get("longitude")
        if latitude is None or longitude is None:
            continue
        for region, anchor in _REGION_ANCHORS.items():
            distance = _haversine_km((float(latitude), float(longitude)), anchor)
            if best is None or distance < best[0]:
                best = (distance, region)
    if best is None:
        return None
    region = best[1]
    return "north" if region == "north_east" else region


def _current_period(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the period of the forecast that covers the present moment."""
    periods = record.get("periods") or []
    if not periods:
        return None
    now = datetime.now(SINGAPORE)
    for period in periods:
        time_period = period.get("timePeriod") or {}
        start = _parse(time_period.get("start"))
        end = _parse(time_period.get("end"))
        if start and end and start <= now < end:
            return period
    return periods[0]


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def get_weather(
    area: Optional[str] = None,
    centres: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return the forecast for the requested area or the given centres.

    The payload always states where the forecast came from and when it was
    read. When the forecast cannot be read the condition is ``unavailable``
    and the limitation says so, so no caller can present a made-up condition
    as a forecast.
    """
    retrieved_at = datetime.now(SINGAPORE).isoformat(timespec="seconds")
    centres = list(centres or [])
    region = _AREAS.get(str(area or "").strip().casefold())
    limitations: list[str] = []
    if area and region is None:
        limitations.append(
            f"The forecast has no region called {area!r}; the island-wide "
            "forecast is shown instead."
        )
    if region is None:
        region = _region_for_centres(centres)
    if area and str(area).strip().casefold() == "north_east":
        limitations.append(
            "The forecast has no north-east region of its own; the north "
            "region is shown."
        )

    record = _fetch()
    if record is None:
        return {
            "condition": "unavailable",
            "condition_code": None,
            "walking_impact": "unknown",
            "requested_area": area,
            "region": region,
            "location": _REGION_PLACES.get(region or ""),
            "centres": centres,
            "island_forecast": None,
            "temperature_celsius": None,
            "valid_period": None,
            "source": SOURCE_NAME,
            "source_key": SOURCE_KEY,
            "source_url": FORECAST_URL,
            "as_of": None,
            "retrieved_at": retrieved_at,
            "mock": False,
            "limitations": limitations + [
                "The NEA weather forecast could not be read, so no forecast "
                "is shown."
            ],
        }

    general = record.get("general") or {}
    general_forecast = general.get("forecast") or {}
    period = _current_period(record)
    regions = (period or {}).get("regions") or {}
    region_forecast = regions.get(region) if region else None
    condition = (region_forecast or {}).get("text") or general_forecast.get("text")
    condition_code = (
        (region_forecast or {}).get("code") or general_forecast.get("code")
    )
    temperature = general.get("temperature") or {}
    valid_period = (general.get("validPeriod") or {}).get("text")
    time_period = (period or {}).get("timePeriod") or {}

    return {
        "condition": condition,
        "condition_code": condition_code,
        "walking_impact": _walking_impact(condition),
        "requested_area": area,
        "region": region,
        "location": _REGION_PLACES.get(region or "") or (
            centres[0] if centres else None
        ),
        "centres": centres,
        "island_forecast": general_forecast.get("text"),
        "temperature_celsius": {
            "high": temperature.get("high"),
            "low": temperature.get("low"),
        },
        "valid_period": valid_period,
        "period": time_period.get("text"),
        "source": SOURCE_NAME,
        "source_key": SOURCE_KEY,
        "source_url": FORECAST_URL,
        "as_of": record.get("updatedTimestamp"),
        "retrieved_at": retrieved_at,
        "mock": False,
        "limitations": limitations,
    }
