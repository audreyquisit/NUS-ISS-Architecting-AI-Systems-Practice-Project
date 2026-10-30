"""OneMap-backed location, geocoding, and route evidence tools."""

from datetime import datetime
from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests


CENTRES_FILE = Path(__file__).resolve().parents[1] / "data" / "hawker_centres.json"
MRT_STATIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "mrt_stations.json"
HTTP_TIMEOUT_SECONDS = float(os.getenv("ONEMAP_TIMEOUT_SECONDS", "12"))
MAX_PLACE_QUERY_CHARS = 200
_ONEMAP_HOST = "www.onemap.gov.sg"
_token: Optional[str] = None
_token_expiry = 0.0
_token_lock = threading.Lock()
_geocode_cache: Dict[str, Dict[str, Any]] = {}
_reverse_geocode_cache: Dict[tuple[float, float], Dict[str, Any]] = {}


class LocationServiceError(RuntimeError):
    """A OneMap or location-catalogue operation could not be completed."""


def _base_url() -> str:
    configured = os.getenv("ONEMAP_BASE_URL", f"https://{_ONEMAP_HOST}").strip()
    try:
        parsed = urlsplit(configured)
        port = parsed.port
    except ValueError as error:
        raise LocationServiceError(
            f"ONEMAP_BASE_URL must use https://{_ONEMAP_HOST}."
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ONEMAP_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise LocationServiceError(
            f"ONEMAP_BASE_URL must use https://{_ONEMAP_HOST}."
        )
    return f"https://{_ONEMAP_HOST}"


def load_hawker_centres() -> List[Dict[str, Any]]:
    with CENTRES_FILE.open(encoding="utf-8") as catalogue:
        return json.load(catalogue)


def canonical_mrt_station_name(place: str) -> Optional[str]:
    """Return the exact station name from the supplied MRT list, if matched."""
    with MRT_STATIONS_FILE.open(encoding="utf-8") as station_file:
        station_names = json.load(station_file)

    def normalise(value: str) -> str:
        value = re.sub(r"\s+(?:mrt\s+)?station$", "", value.strip(), flags=re.I)
        value = re.sub(r"\s+mrt$", "", value, flags=re.I)
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    requested = normalise(place)
    for station_name in station_names:
        if normalise(station_name) == requested:
            return station_name
    return None


def suggest_mrt_station_name(place: str) -> tuple[Optional[str], list[str]]:
    """Return a confident station correction or close alternatives."""
    with MRT_STATIONS_FILE.open(encoding="utf-8") as station_file:
        station_names = json.load(station_file)

    cleaned = re.sub(r"\s+(?:mrt\s+)?station$", "", place.strip(), flags=re.I)
    cleaned = re.sub(r"\s+mrt$", "", cleaned, flags=re.I)

    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    requested = normalise(cleaned)
    # Restrict fuzzy matching to short, station-like inputs.
    if not requested or len(cleaned.split()) > 2 or len(requested) < 4:
        return None, []

    scored = []
    for station in station_names:
        candidate = normalise(station)
        if abs(len(candidate) - len(requested)) > 2:
            continue
        scored.append((SequenceMatcher(None, requested, candidate).ratio(), station))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return None, []

    best_score = scored[0][0]
    close = [name for score, name in scored if score >= 0.78 and best_score - score < 0.06]
    if best_score >= 0.88 and len(close) == 1:
        return scored[0][1], []
    if best_score >= 0.78 and close:
        return None, close[:3]
    return None, []


def _get_access_token(request_id: Optional[str] = None) -> str:
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
        from services.activity_log import log_activity

        auth_url = f"{_base_url()}/api/auth/post/getToken"
        log_activity(
            request_id, "onemap.auth.request.started",
            method="POST",
            host=urlsplit(auth_url).netloc,
            endpoint=urlsplit(auth_url).path,
            body_fields={"email": "[REDACTED]", "password": "[REDACTED]"},
            credentials_logged=False,
        )
        try:
            response = requests.post(
                auth_url,
                json={"email": email, "password": password},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as error:
            log_activity(
            request_id, "onemap.auth.request.failed",
            host=urlsplit(auth_url).netloc,
            endpoint=urlsplit(auth_url).path,
                error_type=type(error).__name__,
                credentials_logged=False,
            )
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
        log_activity(
            request_id, "onemap.auth.request.completed",
            host=urlsplit(auth_url).netloc,
            endpoint=urlsplit(auth_url).path,
            http_status=response.status_code,
            access_token_logged=False,
        )
        return token


def _get(
    url: str, *, params: Dict[str, Any], request_id: Optional[str] = None
) -> Dict[str, Any]:
    from services.activity_log import log_activity

    endpoint = urlsplit(url).path
    # Route calls are logged as one batch by the Location agent so its
    # terminal trace does not print every candidate destination coordinate.
    log_individual_http = endpoint != "/api/public/routingsvc/route"
    safe_params = {}
    for key, value in params.items():
        lowered = key.casefold()
        if lowered == "searchval":
            query = str(value)
            looks_like_address = bool(
                re.search(r"\d", query)
                or re.search(
                    r"\b(?:road|street|avenue|ave|drive|lane|block|blk|unit)\b|#",
                    query,
                    re.I,
                )
            )
            safe_params[key] = "[REDACTED_PRECISE_PLACE]" if looks_like_address else query[:120]
        elif any(secret in lowered for secret in (
            "token", "password", "email", "authorization", "secret",
            "lat", "lng", "location", "coordinate",
        )):
            safe_params[key] = "[REDACTED]"
        else:
            safe_params[key] = value
    access_token = _get_access_token(request_id)
    started = time.monotonic()
    if log_individual_http:
        log_activity(
            request_id, "onemap.http.request.started",
            method="GET",
            host=urlsplit(url).netloc,
            endpoint=endpoint,
            params=safe_params,
            authorization_header_logged=False,
        )
    try:
        response = requests.get(
            url,
            params=params,
            headers={"Authorization": access_token},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        if log_individual_http:
            log_activity(
                request_id, "onemap.http.request.completed",
                method="GET",
                host=urlsplit(url).netloc,
                endpoint=endpoint,
                http_status=response.status_code,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        return data
    except requests.RequestException as error:
        if log_individual_http:
            log_activity(
                request_id, "onemap.http.request.failed",
                method="GET",
                host=urlsplit(url).netloc,
                endpoint=endpoint,
                http_status=(error.response.status_code if error.response else None),
                error_type=type(error).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        raise LocationServiceError("OneMap did not return location evidence.") from error
    except ValueError as error:
        if log_individual_http:
            log_activity(
                request_id, "onemap.http.request.failed",
                method="GET",
                host=urlsplit(url).netloc,
                endpoint=endpoint,
                error_type=type(error).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        raise LocationServiceError("OneMap returned an unreadable response.") from error


def geocode_place(
    place: str, *, request_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Resolve a Singapore place/address to coordinates using OneMap Search."""
    if not isinstance(place, str):
        raise LocationServiceError("Place search must be text.")
    place = place.strip()
    if not place:
        raise LocationServiceError("Place search cannot be empty.")
    if len(place) > MAX_PLACE_QUERY_CHARS:
        raise LocationServiceError(
            f"Place search must be {MAX_PLACE_QUERY_CHARS} characters or fewer."
        )
    key = place.casefold()
    if key in _geocode_cache:
        from services.activity_log import log_activity

        log_activity(
            request_id, "onemap.search.cache_hit",
            query_logged=False,
            api_call_made=False,
        )
        return dict(_geocode_cache[key])
    data = _get(
        f"{_base_url()}/api/common/elastic/search",
        params={"searchVal": place, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1},
        request_id=request_id,
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


def _reverse_geocode_data(
    point: Dict[str, float], *, request_id: Optional[str] = None
) -> Dict[str, Any]:
    cache_key = (round(float(point["lat"]), 6), round(float(point["lng"]), 6))
    if cache_key in _reverse_geocode_cache:
        return _reverse_geocode_cache[cache_key]
    data = _get(
        f"{_base_url()}/api/public/revgeocode",
        params={
            "location": f"{point['lat']},{point['lng']}",
            "buffer": 250,
            "addressType": "All",
        },
        request_id=request_id,
    )
    _reverse_geocode_cache[cache_key] = data
    return data


def _human_address(value: Any) -> Optional[str]:
    if not value or str(value).strip().casefold() in {"nil", "none", "null"}:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = text.title()
    for acronym in ("Mrt", "Hdb", "Cbd", "Ntu", "Nus", "Sbs", "Lta"):
        text = re.sub(rf"\b{acronym}\b", acronym.upper(), text)
    return text


def reverse_geocode_location(
    point: Dict[str, float], *, request_id: Optional[str] = None
) -> Optional[str]:
    """Return the nearest useful address or named place for route-origin display."""
    data = _reverse_geocode_data(point, request_id=request_id)
    entries = data.get("GeocodeInfo") or []
    if not entries:
        return None
    nearest = entries[0]
    building = _human_address(nearest.get("BUILDINGNAME"))
    block = _human_address(nearest.get("BLOCK"))
    road = _human_address(nearest.get("ROAD"))
    if building and "mrt" in building.casefold():
        return building
    if block and road:
        address = f"Blk {block} {road}"
        return f"{building}, {address}" if building else address
    if building:
        return building
    if road:
        return road
    return None


def is_singapore_location(
    point: Dict[str, float], *, request_id: Optional[str] = None
) -> bool:
    """Check that coordinates resolve to a nearby Singapore address in OneMap."""
    return bool(_reverse_geocode_data(point, request_id=request_id).get("GeocodeInfo"))


def get_candidate_centres(
    *, request_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load NEA centre points; actual routes determine travel convenience."""
    results = []
    for centre in load_hawker_centres():
        latitude = centre.get("latitude")
        longitude = centre.get("longitude")
        coordinate_source = "nea_geospatial"
        if latitude is None or longitude is None:
            point = geocode_place(
                centre.get("search_query") or centre["name"],
                request_id=request_id,
            )
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
    start: Dict[str, float], end: Dict[str, float], mode: str, *,
    num_itineraries: int = 3, request_id: Optional[str] = None,
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
    data = _get(
        f"{_base_url()}/api/public/routingsvc/route",
        params=params,
        request_id=request_id,
    )
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
        legs = fastest.get("legs")
        if not isinstance(legs, list) or any(not isinstance(leg, dict) for leg in legs):
            legs = []  # Duration may be usable even when leg evidence is absent.
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


def _transfer_count(legs: List[Dict[str, Any]]) -> Optional[int]:
    if not legs or any(
        not isinstance(leg, dict) or not isinstance(leg.get("mode"), str)
        or not leg["mode"].strip() for leg in legs
    ):
        return None
    transit_legs = [
        leg for leg in legs
        if leg["mode"].strip().casefold() not in {"walk", "walking"}
    ]
    return max(0, len(transit_legs) - 1)
