"""Build reproducible synthetic queue records from the local mock catalogue."""

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_FILE = DATA_DIR / "hawker_centres.json"
STALLS_FILE = DATA_DIR / "hawker_stall.json"
MOCK_STALLS_FILE = DATA_DIR / "hawker_stall_mock.json"
QUEUE_SOURCE = "synthetic_queue_mock"
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
    mock_payload = json.loads(MOCK_STALLS_FILE.read_text(encoding="utf-8"))
    centres_by_id = {centre["id"]: centre for centre in centres}
    records = []

    stalls_by_id = {
        stall.get("stall_id"): stall
        for stall in stalls_payload.get("stalls", [])
        if stall.get("stall_id")
    }
    stalls_by_id.update({
        stall.get("stall_id"): stall
        for stall in mock_payload.get("stalls", [])
        if stall.get("stall_id")
    })

    for stall in stalls_by_id.values():
        centre = centres_by_id.get(stall.get("hawker_centre_id"))
        stall_id = stall.get("stall_id")
        stall_name = str(stall.get("name") or "").strip()
        if not centre or not stall_id or not stall_name:
            continue

        base_wait, serving_time = _queue_values(stall_id)
        queue_score = max(
            0,
            min(100, round(100 - (base_wait * 3) - (serving_time * 2))),
        )
        crowd = _crowd_level(base_wait)
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
            "queue_is_mock": True,
            "queue_mock_notice": QUEUE_NOTICE,
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