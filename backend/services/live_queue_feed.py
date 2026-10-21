"""Build reproducible synthetic queue records from the local mock catalogue.

The queue values are simulated. They are derived from a stable hash of the
stall ID so the same stall always shows the same estimate, which keeps the
assistant honest about repeating itself. A live signal from LTA DataMall may
be added as area context when an account key is configured, but it never
replaces the simulated estimate and is never presented as an observation of
the stall.
"""

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from services import lta_datamall


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_FILE = DATA_DIR / "hawker_centres.json"
STALLS_FILE = DATA_DIR / "hawker_stall.json"
MOCK_STALLS_FILE = DATA_DIR / "hawker_stall_mock.json"
QUEUE_SOURCE = "simulated_queue_estimate"
QUEUE_NOTICE = (
    "Simulated queue and crowd estimate generated from a stable stall ID; "
    "not a live or observed queue."
)

REGION_ALIASES = {
    "central": ("central", "raffles", "marina", "city", "bugis", "orchard", "cbd", "tanjong", "clarke quay"),
    "east": ("east", "bedok", "tampines", "marine parade", "eunos", "paya lebar"),
    "west": ("west", "jurong", "clementi", "boon lay", "choa chu kang", "bukit batok"),
    "north": ("north", "woodlands", "yishun", "sembawang", "admiralty", "khatib"),
    "north_east": ("north-east", "northeast", "hougang", "serangoon", "punggol", "sengkang"),
    "south": ("south", "queenstown", "buona vista", "pasir panjang", "dunman", "katong"),
}

REGION_ANCHORS = {
    "west": (1.335, 103.705),
    "east": (1.350, 103.955),
    "north": (1.435, 103.790),
    "north_east": (1.370, 103.895),
    "south": (1.265, 103.820),
    "central": (1.300, 103.840),
}


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _queue_values(stall_id: str) -> tuple[int, int]:
    """Derive stable demo values from a source stall ID, never Python's hash()."""
    digest = hashlib.blake2b(
        f"{stall_id}:queue-v1".encode("utf-8"), digest_size=4
    ).digest()
    value = int.from_bytes(digest, byteorder="big")
    return 4 + value % 22, 4 + (value // 22) % 7


def _crowd_level(queue_minutes: int) -> str:
    if queue_minutes <= 5:
        return "Low"
    if queue_minutes <= 15:
        return "Moderate"
    if queue_minutes <= 25:
        return "High"
    return "Very High"


def _region_for_centre(centre: dict[str, Any]) -> str:
    latitude = float(centre["latitude"])
    longitude = float(centre["longitude"])
    return min(
        REGION_ANCHORS,
        key=lambda region: _haversine_km(
            latitude,
            longitude,
            REGION_ANCHORS[region][0],
            REGION_ANCHORS[region][1],
        ),
    )


@lru_cache(maxsize=1)
def _mrt_station_names() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "data" / "mrt_stations.json"
    if not path.exists():
        return ()
    names = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(name) for name in names)


@lru_cache(maxsize=1)
def _station_codes() -> dict[str, str]:
    """Return the DataMall station codes, when the service is reachable."""
    if not lta_datamall.is_enabled():
        return {}
    stations = lta_datamall.get_train_stations() or []
    return {
        str(station.get("station_name") or "").strip().casefold():
        str(station.get("station_code") or "").strip()
        for station in stations
        if station.get("station_name") and station.get("station_code")
    }


def _live_station_context(centre: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the live crowd context of the station near a centre.

    The context is area information from LTA DataMall. It says nothing about
    the queue at a stall, so it is kept apart from the queue estimate.
    """
    if not lta_datamall.is_enabled():
        return None
    codes = _station_codes()
    if not codes:
        return None
    text = _normalise(" ".join(
        str(centre.get(key) or "")
        for key in ("name", "official_name", "address", "street_name")
    ))
    if not text:
        return None
    matches = [
        code for name, code in codes.items()
        if len(name) >= 4 and f" {name} " in f" {text} "
    ]
    if len(matches) != 1:
        return None
    crowd = lta_datamall.get_station_crowd_density(matches[0])
    if not crowd:
        return None
    return {
        "station_code": crowd.get("station_code"),
        "crowd_density": crowd.get("crowd_density"),
        "crowd_density_label": crowd.get("crowd_density_label"),
        "source": lta_datamall.SOURCE_NAME,
        "source_key": lta_datamall.SOURCE_KEY,
        "is_live": True,
        "note": (
            "Live station crowd density for the area. It is not a queue "
            "measurement for this stall."
        ),
    }


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(value))


@lru_cache(maxsize=1)
def _all_mock_queue_records() -> tuple[dict[str, Any], ...]:
    centres = json.loads(CENTRES_FILE.read_text(encoding="utf-8"))
    stalls_payload = json.loads(STALLS_FILE.read_text(encoding="utf-8"))
    centres_by_id = {centre["id"]: centre for centre in centres}
    records = []

    stalls_by_id = {
        stall.get("stall_id"): stall
        for stall in stalls_payload.get("stalls", [])
        if stall.get("stall_id")
    }

    for stall in stalls_by_id.values():
        centre_id = stall.get("hawker_centre_id")
        centre = centres_by_id.get(centre_id) if centre_id else None
        stall_id = stall.get("stall_id")
        stall_name = str(stall.get("name") or "").strip()
        if not centre or not stall_id or not stall_name:
            # Fall back to matching by hawker_centre text name against
            # catalogue centre names (e.g. Wak-Wak stalls with no
            # hawker_centre_id linking them to the NEA catalogue).
            wanted_name = _normalise(stall.get("hawker_centre") or "")
            if wanted_name:
                for c in centres_by_id.values():
                    if _normalise(c.get("name")) == wanted_name or _normalise(
                        c.get("official_name", "")
                    ) == wanted_name:
                        centre = c
                        centre_id = c["id"]
                        break
            if not centre:
                continue

        base_wait, serving_time = _queue_values(stall_id)
        queue_score = max(
            0,
            min(100, round(100 - (base_wait * 3) - (serving_time * 2))),
        )
        crowd = _crowd_level(base_wait)
        live_context = _live_station_context(centre)
        records.append({
            "stall_id": stall_id,
            "stall_name": stall_name,
            "hawker_centre_id": centre["id"],
            "hawker_centre": centre["name"],
            "hawker_centre_source_name": stall.get("hawker_centre", ""),
            "region": _region_for_centre(centre),
            "base_queue_minutes": base_wait,
            "queue_minutes": base_wait,
            "crowd_level": crowd,
            "wait_band": (
                "Short" if base_wait <= 5
                else "Moderate" if base_wait <= 15
                else "Long"
            ),
            "estimated_service_time_minutes": serving_time,
            "queue_score": queue_score,
            "source": QUEUE_SOURCE,
            "queue_is_simulated": True,
            "queue_estimate_notice": QUEUE_NOTICE,
            "live_area_context": live_context,
            "is_mock": bool(stall.get("is_mock", False)),
            "data_origin": stall.get(
                "data_origin",
                "synthetic_mock" if stall.get("is_mock") else "hawker_stall_catalogue",
            ),
            "stall_mock_notice": stall.get("mock_notice"),
        })

    return tuple(records)


def _requested_region(task: str) -> Optional[str]:
    text = _normalise(task)
    matches = [
        (len(_normalise(alias)), region)
        for region, aliases in REGION_ALIASES.items()
        for alias in aliases
        if f" {_normalise(alias)} " in f" {text} "
    ]
    return max(matches)[1] if matches else None


def _requested_centre_ids(task: str) -> set[str]:
    text = _normalise(task)
    if not text:
        return set()
    centres = json.loads(CENTRES_FILE.read_text(encoding="utf-8"))
    names_by_id: dict[str, set[str]] = {}
    for centre in centres:
        names_by_id[centre["id"]] = {
            centre.get("name", ""),
            centre.get("official_name", ""),
            *centre.get("aliases", []),
        }
    mock_payload = json.loads(MOCK_STALLS_FILE.read_text(encoding="utf-8"))
    for stall in mock_payload.get("stalls", []):
        centre_id = stall.get("hawker_centre_id")
        source_name = stall.get("hawker_centre")
        if centre_id in names_by_id and source_name:
            names_by_id[centre_id].add(source_name)

    matches = []
    padded_text = f" {text} "
    for centre_id, names in names_by_id.items():
        for name in names:
            normalized_name = _normalise(name)
            if (
                normalized_name
                and len(normalized_name.split()) >= 2
                and f" {normalized_name} " in padded_text
            ):
                matches.append((len(normalized_name), centre_id))
    if not matches:
        return set()
    longest_match = max(length for length, _ in matches)
    return {
        centre_id for length, centre_id in matches
        if length == longest_match
    }


def get_live_queue_feed(task: Optional[str] = None) -> List[dict]:
    """Return simulated queue stats for catalogue stalls, filtered by centre or region."""
    records = list(_all_mock_queue_records())
    query = task or ""
    centre_ids = _requested_centre_ids(query)
    if centre_ids:
        records = [
            record for record in records
            if record["hawker_centre_id"] in centre_ids
        ]
    else:
        region = _requested_region(query)
        if region:
            records = [record for record in records if record["region"] == region]

    return [dict(record) for record in records]