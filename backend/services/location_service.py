"""OneMap-backed location, geocoding, and route evidence tools."""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests


CENTRES_FILE = Path(__file__).resolve().parents[1] / "data" / "hawker_centres.json"
HTTP_TIMEOUT_SECONDS = float(os.getenv("ONEMAP_TIMEOUT_SECONDS", "12"))
_token: Optional[str] = None
_token_expiry = 0.0
_token_lock = threading.Lock()
_geocode_cache: Dict[str, Dict[str, Any]] = {}


class LocationServiceError(RuntimeError):
    """A OneMap or location-catalogue operation could not be completed."""


def _base_url() -> str:
    return os.getenv("ONEMAP_BASE_URL", "https://www.onemap.gov.sg").rstrip("/")


def load_hawker_centres() -> List[Dict[str, Any]]:
    with CENTRES_FILE.open(encoding="utf-8") as catalogue:
        return json.load(catalogue)


def _get_access_token() -> str:
    global _token, _token_expiry
    if _token and time.time() < _token_expiry:
        return _token

    with _token_lock:
        if _token and time.time() < _token_expiry:
            return _token
        email = os.getenv("ONEMAP_EMAIL")
        password = os.getenv("ONEMAP_EMAIL_PASSWORD")
        if not email or not password:
            raise LocationServiceError(
                "OneMap is not configured. Set ONEMAP_EMAIL and ONEMAP_EMAIL_PASSWORD."
            )
        try:
            response = requests.post(
                f"{_base_url()}/api/auth/post/getToken",
                json={"email": email, "password": password},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as error:
            raise LocationServiceError("Could not authenticate with OneMap.") from error

        token = body.get("access_token")
        if not token:
            raise LocationServiceError("OneMap did not return an access token.")
        expiry_timestamp = body.get("expiry_timestamp")
        try:
            expires_at = float(expiry_timestamp)
        except (TypeError, ValueError):
            # The documented token lifetime is 3 days. Refresh a little early.
            expires_at = time.time() + 3 * 24 * 60 * 60
        _token = token
        _token_expiry = max(time.time(), expires_at - 60)
        return token


def _get(url: str, *, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"Authorization": _get_access_token()},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise LocationServiceError("OneMap did not return location evidence.") from error
    except ValueError as error:
        raise LocationServiceError("OneMap returned an unreadable response.") from error


def geocode_place(place: str) -> Optional[Dict[str, Any]]:
    """Resolve a Singapore place/address to coordinates using OneMap Search."""
    key = place.casefold().strip()
    if key in _geocode_cache:
        return dict(_geocode_cache[key])
    data = _get(
        f"{_base_url()}/api/common/elastic/search",
        params={"searchVal": place, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1},
    )
    for result in data.get("results", []):
        try:
            point = {
                "lat": float(result["LATITUDE"]),
                "lng": float(result["LONGITUDE"]),
                "label": result.get("BUILDING") or result.get("ADDRESS") or place,
                "source": "onemap_search",
            }
        except (KeyError, TypeError, ValueError):
            continue
        _geocode_cache[key] = point
        return dict(point)
    return None


def get_candidate_centres() -> List[Dict[str, Any]]:
    """Load NEA centre points; actual routes determine travel convenience."""
    results = []
    for centre in load_hawker_centres():
        latitude = centre.get("latitude")
        longitude = centre.get("longitude")
        coordinate_source = "nea_geospatial"
        if latitude is None or longitude is None:
            point = geocode_place(centre.get("search_query") or centre["name"])
            if point is None:
                continue
            latitude, longitude = point["lat"], point["lng"]
            coordinate_source = point["source"]
        results.append({
            **centre,
            "latitude": latitude,
            "longitude": longitude,
            "coordinate_source": coordinate_source,
        })
    return results


def get_route(
    start: Dict[str, float], end: Dict[str, float], mode: str, *, num_itineraries: int = 3
) -> Dict[str, Any]:
    """Return OneMap's fastest returned itinerary for the selected mode."""
    route_type = "pt" if mode == "pt" else mode
    params: Dict[str, Any] = {
        "start": f"{start['lat']},{start['lng']}",
        "end": f"{end['lat']},{end['lng']}",
        "routeType": route_type,
    }
    if route_type == "pt":
        now = datetime.now(ZoneInfo("Asia/Singapore"))
        params.update({
            "date": now.strftime("%m-%d-%Y"),
            "time": now.strftime("%H:%M:%S"),
            "mode": "transit",
            "numItineraries": min(max(num_itineraries, 1), 3),
        })
    data = _get(f"{_base_url()}/api/public/routingsvc/route", params=params)
    summary = data.get("route_summary") or {}
    total_seconds = _number(summary.get("total_time"))
    walking_distance = _number(summary.get("walking_distance"))

    # PT responses usually wrap itineraries in plan.itineraries; non-PT routes
    # expose route_summary directly. Retain a defensive parser for API variants.
    itineraries = (data.get("plan") or {}).get("itineraries") or data.get("itineraries") or []
    if itineraries:
        usable_itineraries = [
            item for item in itineraries if _duration_seconds(item) is not None
        ]
        if not usable_itineraries:
            raise LocationServiceError("OneMap returned itineraries without travel durations.")
        fastest = min(usable_itineraries, key=lambda item: _duration_seconds(item))
        total_seconds = _duration_seconds(fastest)
        legs = fastest.get("legs", [])
        if walking_distance is None:
            walking_distance = sum(
                _number(leg.get("distance")) or 0
                for leg in legs
                if "walk" in str(leg.get("mode", "")).casefold()
            ) or None
        summary_text = _summarize_legs(legs)
    else:
        legs = []
        if total_seconds is not None:
            # Non-PT route_summary.total_time is expressed in minutes.
            total_seconds *= 60
        summary_text = None

    if total_seconds is None:
        raise LocationServiceError("OneMap returned no usable travel duration for this route.")
    return {
        "total_duration_min": max(1, math.ceil(total_seconds / 60)),
        "walking_distance_m": walking_distance,
        "mode": mode,
        "source": "onemap",
        "confidence": 0.85,
        "transfers": _transfer_count(legs) if itineraries else None,
        "limitation": None,
        "itinerary_summary": summary_text,
    }


def _number(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _duration_seconds(itinerary: Dict[str, Any]) -> Optional[float]:
    duration = _number(itinerary.get("duration"))
    if duration is None:
        duration = _number((itinerary.get("route_summary") or {}).get("total_time"))
    if duration is None:
        return None
    # OneMap PT itinerary `duration` is seconds, while route_summary.total_time
    # is expressed in minutes.
    if itinerary.get("duration") is None:
        return duration * 60
    unit = str(itinerary.get("duration_unit", "seconds")).casefold()
    return duration * 60 if unit in {"minute", "minutes", "min"} else duration


def _summarize_legs(legs: List[Dict[str, Any]]) -> Optional[str]:
    modes = []
    for leg in legs:
        mode = leg.get("mode")
        route = leg.get("route") or leg.get("routeLongName") or leg.get("routeShortName")
        if mode:
            label = str(mode).replace("_", " ").title()
            if route:
                label += f" {route}"
            if label not in modes:
                modes.append(label)
    return " → ".join(modes) or None


def _transfer_count(legs: List[Dict[str, Any]]) -> int:
    transit_legs = [
        leg for leg in legs
        if str(leg.get("mode", "")).casefold() not in {"walk", "walking", ""}
    ]
    return max(0, len(transit_legs) - 1)
