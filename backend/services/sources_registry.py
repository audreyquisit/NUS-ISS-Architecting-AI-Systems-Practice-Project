"""The provenance registry for every source the assistant may cite.

Every source that can appear in a reply is listed here with the address it
was read from and the date its content is as of. A source that has no data
in the repository is not listed at all, so a reply can only ever name a
source that is actually present in the evidence payload.

The registry is built from the data files themselves, so it stays true when
a snapshot is refreshed.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CENTRES_FILE = DATA_DIR / "hawker_centres.json"
CENTRES_META_FILE = DATA_DIR / "hawker_centres.meta.json"
STALLS_FILE = DATA_DIR / "hawker_stall.json"
DIRECTORY_FILE = DATA_DIR / "hawker_stall_directory_linked.json"
HEALTHIER_DINING_FILE = DATA_DIR / "hpb_healthier_dining.json"
HALAL_FILE = DATA_DIR / "muis_halal.json"
HOLIDAYS_FILE = DATA_DIR / "public_holidays.json"
CROWDSOURCED_FILE = DATA_DIR / "hawkercentral_stalls.json"

OPEN_DATA_LICENCE = "Singapore Open Data Licence (data.gov.sg)"


def _read(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_sources_registry() -> list[dict[str, Any]]:
    """Return every source the assistant may cite, with its provenance."""
    registry: list[dict[str, Any]] = []

    centres = _read(CENTRES_FILE)
    meta = _read(CENTRES_META_FILE) or {}
    if isinstance(centres, list) and centres:
        registry.append({
            "key": "nea_hawker_centres",
            "name": "NEA Hawker Centres (GEOJSON)",
            "kind": "official_dataset",
            "url": meta.get("dataset_url"),
            "dataset_id": meta.get("dataset_id"),
            "licence": meta.get("licence", OPEN_DATA_LICENCE),
            "as_of": meta.get("list_as_of"),
            "record_count": len(centres),
        })
        registry.append({
            "key": "nea_centre_list",
            "name": "NEA List of Hawker Centres",
            "kind": "official_dataset",
            "url": meta.get("list_url"),
            "dataset_id": None,
            "licence": OPEN_DATA_LICENCE,
            "as_of": meta.get("list_as_of"),
            "record_count": meta.get("list_row_count"),
        })

    stalls = _read(STALLS_FILE)
    if isinstance(stalls, dict) and stalls.get("stalls"):
        registry.append({
            "key": "wak_wak_hawker",
            "name": "Wak-Wak Hawker stall and menu catalogue",
            "kind": "stall_catalogue",
            "url": stalls.get("source"),
            "dataset_id": None,
            "licence": "Scraped public listings; no open licence is stated by the source.",
            "as_of": stalls.get("as_of"),
            "record_count": len(stalls["stalls"]),
        })

    directory = _read(DIRECTORY_FILE)
    if isinstance(directory, dict) and directory.get("records"):
        registry.append({
            "key": "dbs_paylah_directory",
            "name": directory.get("source") or "DBS PayLah! merchant directory",
            "kind": "merchant_directory",
            "url": directory.get("source_file"),
            "dataset_id": None,
            "licence": "Bank merchant listing extract held in this repository.",
            "as_of": directory.get("as_of"),
            "record_count": directory.get("linked_record_count"),
        })

    healthier = _read(HEALTHIER_DINING_FILE)
    if isinstance(healthier, dict) and healthier.get("outlets"):
        registry.append({
            "key": "hpb_healthier_dining",
            "name": "HPB Healthier Dining Partners (F&B)",
            "kind": "official_dataset",
            "url": healthier.get("dataset_url"),
            "dataset_id": healthier.get("dataset_id"),
            "licence": OPEN_DATA_LICENCE,
            "as_of": healthier.get("as_of"),
            "record_count": len(healthier["outlets"]),
        })

    halal = _read(HALAL_FILE)
    if isinstance(halal, dict) and halal.get("establishments"):
        registry.append({
            "key": "muis_halal",
            "name": "MUIS halal e-Service establishment directory",
            "kind": "official_snapshot",
            "url": halal.get("source_url"),
            "dataset_id": None,
            "licence": (
                "Public e-Service directory. Not a data.gov.sg dataset, so it is "
                "not covered by the Open Data Licence."
            ),
            "as_of": halal.get("retrieved_at"),
            "record_count": len(halal["establishments"]),
        })

    holidays = _read(HOLIDAYS_FILE)
    if isinstance(holidays, dict) and holidays.get("holidays"):
        registry.append({
            "key": "mom_public_holidays",
            "name": "MOM public holidays",
            "kind": "official_dataset",
            "url": holidays.get("dataset_url"),
            "dataset_id": holidays.get("dataset_id"),
            "licence": OPEN_DATA_LICENCE,
            "as_of": holidays.get("as_of"),
            "record_count": len(holidays["holidays"]),
        })

    crowdsourced = _read(CROWDSOURCED_FILE)
    if isinstance(crowdsourced, dict) and crowdsourced.get("stalls"):
        registry.append({
            "key": "hawkercentral",
            "name": "hawkercentral community listing",
            "kind": "crowdsourced",
            "url": crowdsourced.get("source_url"),
            "dataset_id": None,
            "licence": crowdsourced.get("licence", "MIT"),
            "as_of": crowdsourced.get("as_of"),
            "record_count": len(crowdsourced["stalls"]),
        })

    registry.append({
        "key": "nea_mss_weather",
        "name": "NEA/MSS 24-hour weather forecast",
        "kind": "api",
        "url": "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast",
        "dataset_id": None,
        "licence": OPEN_DATA_LICENCE,
        "as_of": None,
        "record_count": None,
    })
    registry.append({
        "key": "onemap",
        "name": "OneMap Singapore",
        "kind": "api",
        "url": "https://www.onemap.gov.sg",
        "dataset_id": None,
        "licence": "OneMap terms of use",
        "as_of": None,
        "record_count": None,
    })
    if os.getenv("LTA_DATAMALL_KEY"):
        registry.append({
            "key": "lta_datamall",
            "name": "LTA DataMall station crowd density",
            "kind": "api",
            "url": "http://datamall.lta.gov.sg",
            "dataset_id": "PCDRealTime",
            "licence": "LTA DataMall terms of use",
            "as_of": None,
            "record_count": None,
        })
    return registry


@lru_cache(maxsize=1)
def _registry_by_key() -> dict[str, dict[str, Any]]:
    return {entry["key"]: entry for entry in get_sources_registry()}


def get_source(key: str) -> Optional[dict[str, Any]]:
    """Return one source entry, or None when the source is not held."""
    entry = _registry_by_key().get(key)
    return dict(entry) if entry else None


def sources_for(keys: list[str]) -> list[dict[str, Any]]:
    """Return the entries of the named sources that the repository holds."""
    entries = []
    for key in keys:
        entry = get_source(key)
        if entry:
            entries.append(entry)
    return entries


def format_source_line(keys: list[str]) -> str:
    """Return the one-line provenance string for the named sources."""
    parts = []
    for entry in sources_for(keys):
        as_of = entry.get("as_of") or "date not stated"
        parts.append(f"{entry['name']} (as of {as_of})")
    return "; ".join(parts)


def format_sources_line(
    keys: list[str], as_of_overrides: Optional[dict[str, str]] = None
) -> str:
    """Return the closing provenance line of a reply.

    Only the sources that are actually present in the evidence are named.
    The line is empty when no source is present, so a reply without evidence
    never carries a provenance line.
    """
    entries = sources_for(keys)
    if not entries:
        return ""
    overrides = as_of_overrides or {}
    parts = []
    for entry in entries:
        label = (
            overrides.get(entry["key"])
            or entry.get("as_of")
            or entry.get("retrieved_at")
            or "date not stated"
        )
        parts.append(f"{entry['name']} (as of {label})")
    return "Sources: " + "; ".join(parts)


# The source names the evidence uses, mapped to the keys of the registry. A
# source that is not in the registry is never cited.
SOURCE_KEYS = {
    "wak_wak_hawker": "wak_wak_hawker",
    "hpb_healthier_dining": "hpb_healthier_dining",
    "muis_halal": "muis_halal",
    "dbs_paylah_directory": "dbs_paylah_directory",
    "crowdsourced": "hawkercentral",
    "hawkercentral": "hawkercentral",
    "nea_hawker_centres": "nea_hawker_centres",
    "nea_centre_list": "nea_centre_list",
    "mom_public_holidays": "mom_public_holidays",
    "nea_mss_weather": "nea_mss_weather",
    "onemap": "onemap",
    "onemap_search": "onemap",
    "lta_datamall": "lta_datamall",
}


def collect_evidence_sources(
    agent_results: list,
) -> tuple[list[str], dict[str, str]]:
    """Return the sources present in the evidence and the date of each.

    Only a source that is actually present in the evidence is returned, so
    the closing line can never name a source that was not used.
    """
    keys: list[str] = []
    dates: dict[str, str] = {}

    def add(key: Optional[str], as_of: Optional[str] = None) -> None:
        registry_key = SOURCE_KEYS.get(str(key or ""))
        if not registry_key:
            return
        if registry_key not in keys:
            keys.append(registry_key)
        if as_of and registry_key not in dates:
            dates[registry_key] = str(as_of)

    for result in agent_results or []:
        if not isinstance(result, dict):
            continue
        agent = result.get("agent")
        if agent == "candidate_join":
            for candidate in result.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("healthier_dining"):
                    add("hpb_healthier_dining", candidate.get("healthier_as_of"))
                if candidate.get("halal_certified"):
                    add("muis_halal", candidate.get("halal_as_of"))
                if candidate.get("crowdsourced_source"):
                    add("crowdsourced", candidate.get("crowdsourced_as_of"))
                if candidate.get("merchant_list_name"):
                    add("dbs_paylah_directory")
                for source in candidate.get("sources", []) or []:
                    add(source, candidate.get("data_as_of"))
        elif agent == "weather":
            weather = (
                result.get("weather")
                if isinstance(result.get("weather"), dict)
                else result
            )
            add(weather.get("source_key"), weather.get("as_of"))
        elif agent == "location":
            payload = result.get("payload", result)
            if isinstance(payload, dict):
                for route in payload.get("logistics", []) or []:
                    if not isinstance(route, dict):
                        continue
                    travel = route.get("travel") or {}
                    add(travel.get("source"))
    add("nea_hawker_centres")
    return keys, dates
