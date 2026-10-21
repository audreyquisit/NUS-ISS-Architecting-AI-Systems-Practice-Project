"""Refresh the hawker-centre catalogue from the official NEA sources.

Two official sources are combined:

* the NEA *Hawker Centres (GEOJSON)* dataset on data.gov.sg, which carries the
  full official attribute set for every centre; and
* the NEA *List of Hawker Centres* PDF, which is the authoritative centre list
  and names the entity that manages each centre's common property.

The refresh is idempotent: re-running it reproduces the same catalogue. The
stable centre ``id``, user-facing ``name`` and ``aliases`` of every existing
centre are preserved so stall links and stored sessions keep working; a centre
whose official name changed keeps its previous official name as an alias.

Usage (from backend/):
    python scripts/refresh_hawker_centres.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nea_pdf import extract_centre_rows  # noqa: E402
from services.source_matching import (  # noqa: E402
    extract_postcode, extract_postcodes, normalise_name,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"
CENTRES_META_PATH = DATA_DIR / "hawker_centres.meta.json"

NEA_GEOJSON_DATASET_ID = "d_4a086da0a5553be1d89383cd90d07ecd"
NEA_GEOJSON_DATASET_URL = (
    "https://data.gov.sg/datasets/" + NEA_GEOJSON_DATASET_ID + "/view"
)
NEA_DOWNLOAD_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    + NEA_GEOJSON_DATASET_ID
    + "/poll-download"
)
NEA_LICENCE = "Singapore Open Data Licence (data.gov.sg)"
NEA_SOURCE_NAME = "NEA Hawker Centres (data.gov.sg geospatial dataset)"

# The NEA list PDF is dated in its file name. The July 2026 list is the
# current one; the 2025 copy is kept as a fallback.
NEA_LIST_URLS = (
    "https://www.nea.gov.sg/docs/default-source/hawker-centres-documents/"
    "list-of-hcs_-23-july-2026.pdf",
    "https://www.nea.gov.sg/docs/default-source/hawker-centres-documents/"
    "list-of-hcs_-25-july-2025.pdf",
)
NEA_LIST_URL = NEA_LIST_URLS[0]
NEA_LIST_AS_OF = "2026-07-23"

# NEA statuses that describe a centre the public can use.
USABLE_STATUSES = (
    "Existing",
    "Existing (new)",
    "Existing (replacement)",
    "Interim Centre",
)

HTTP_TIMEOUT_SECONDS = 60


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def download_geojson() -> list[dict[str, Any]]:
    """Download the NEA hawker-centre GEOJSON and return its properties."""
    print(f"Downloading the NEA hawker-centre dataset {NEA_GEOJSON_DATASET_ID}...")
    poll = requests.get(NEA_DOWNLOAD_URL, timeout=HTTP_TIMEOUT_SECONDS)
    poll.raise_for_status()
    payload = poll.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"NEA dataset download failed: {payload.get('errMsg')}")
    body = requests.get(payload["data"]["url"], timeout=HTTP_TIMEOUT_SECONDS)
    body.raise_for_status()
    features = json.loads(body.text).get("features", [])
    if not features:
        raise RuntimeError("The NEA dataset returned no centre features.")
    print(f"  {len(features)} centre records")
    return features


def download_centre_list() -> list[dict[str, Any]]:
    """Download and read the NEA hawker-centre list PDF."""
    last_error: Optional[Exception] = None
    for url in NEA_LIST_URLS:
        try:
            print(f"Downloading the NEA hawker-centre list from {url}...")
            rows = extract_centre_rows(_download(url))
            print(f"  {len(rows)} centre rows")
            return rows
        except Exception as error:  # noqa: BLE001 - report the last failure
            last_error = error
            print(f"  unavailable: {error}")
    raise RuntimeError(f"The NEA hawker-centre list could not be downloaded: {last_error}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _split_locators(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split("/") if part.strip()]


def _centre_from_feature(feature: dict[str, Any]) -> dict[str, Any]:
    """Build a catalogue record from an NEA GEOJSON feature."""
    properties = feature.get("properties", {}) or {}
    geometry = feature.get("geometry", {}) or {}
    coordinates = geometry.get("coordinates") or []
    longitude = coordinates[0] if len(coordinates) > 0 else None
    latitude = coordinates[1] if len(coordinates) > 1 else None
    address = str(properties.get("ADDRESS_MYENV") or "").strip()
    postal_code = str(properties.get("ADDRESSPOSTALCODE") or "").strip()
    if not address and postal_code:
        address = f"Singapore {postal_code}"
    return {
        "id": _slug(str(properties.get("NAME") or "")),
        "name": str(properties.get("NAME") or "").strip(),
        "official_name": str(properties.get("NAME") or "").strip(),
        "aliases": [],
        "address": address,
        "search_query": address,
        "latitude": latitude,
        "longitude": longitude,
        "status": str(properties.get("STATUS") or "").strip(),
        "cooked_food_stalls": properties.get("NUMBER_OF_COOKED_FOOD_STALLS"),
        "nea_object_id": properties.get("OBJECTID"),
        "official_description": (properties.get("DESCRIPTION") or None),
        "heritage_year": (properties.get("EST_ORIGINAL_COMPLETION_DATE") or None),
        "hup_completion_date": (properties.get("HUP_COMPLETION_DATE") or None),
        "photo_url": (properties.get("PHOTOURL") or None),
        "co_locators": _split_locators(properties.get("INFO_ON_CO_LOCATORS")),
        "postal_code": postal_code or None,
        "building_name": (properties.get("ADDRESSBUILDINGNAME") or None),
        "street_name": (properties.get("ADDRESSSTREETNAME") or None),
        "block_number": (properties.get("ADDRESSBLOCKHOUSENUMBER") or None),
        "myenv_address": (properties.get("ADDRESS_MYENV") or None),
        "awarded_date": (properties.get("AWARDED_DATE") or None),
        "implementation_date": (properties.get("IMPLEMENTATION_DATE") or None),
        "nea_record_updated": (properties.get("FMEL_UPD_D") or None),
        "source_as_of": NEA_LIST_AS_OF,
    }


def _apply_feature(centre: dict[str, Any], feature: dict[str, Any]) -> None:
    """Copy the official attributes of an NEA feature onto a centre record."""
    properties = feature.get("properties", {}) or {}
    updated = _centre_from_feature(feature)
    for key, value in updated.items():
        if key in {"id", "name", "aliases"}:
            continue
        if value is None and key in {"latitude", "longitude"}:
            continue
        if key in {"address", "search_query"} and not properties.get("ADDRESS_MYENV"):
            # The NEA record has no address for this centre; keep the one the
            # catalogue already holds rather than degrading it.
            continue
        centre[key] = value


def _match_list_rows(
    centres: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Link the official centre list to catalogue centres by postcode.

    A link is only accepted when the postcode identifies exactly one centre,
    or when the postcode and the official name agree on exactly one centre.
    """
    by_postcode: dict[str, list[dict[str, Any]]] = {}
    for centre in centres:
        for postcode in extract_postcodes(centre.get("address", "")):
            by_postcode.setdefault(postcode, []).append(centre)

    linked: dict[int, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    for row in rows:
        postcodes = extract_postcodes(row.get("address", ""))
        candidates: list[dict[str, Any]] = []
        for postcode in postcodes:
            for centre in by_postcode.get(postcode, []):
                if centre not in candidates:
                    candidates.append(centre)
        if len(candidates) > 1:
            wanted = normalise_name(row.get("name"))
            narrowed = [
                centre for centre in candidates
                if wanted and (
                    wanted == normalise_name(centre.get("name"))
                    or wanted == normalise_name(centre.get("official_name"))
                    or any(
                        wanted == normalise_name(alias)
                        for alias in centre.get("aliases", []) or []
                    )
                )
            ]
            if narrowed:
                candidates = narrowed
        if len(candidates) != 1 and row.get("name"):
            # The list sometimes records a different postcode than the
            # geospatial data; a unique name still identifies the centre.
            wanted = normalise_name(row.get("name"))
            by_name = [
                centre for centre in centres
                if wanted and (
                    wanted == normalise_name(centre.get("name"))
                    or wanted == normalise_name(centre.get("official_name"))
                    or any(
                        wanted == normalise_name(alias)
                        for alias in centre.get("aliases", []) or []
                    )
                )
            ]
            if len(by_name) == 1:
                candidates = by_name
        if len(candidates) == 1:
            linked[id(row)] = candidates[0]
        else:
            unmatched.append(row)
    return linked, unmatched


def build_catalogue(
    features: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge the official attributes and the centre list into the catalogue."""
    centres = [dict(centre) for centre in existing]
    usable_object_ids = {
        (feature.get("properties", {}) or {}).get("OBJECTID")
        for feature in features
        if str((feature.get("properties", {}) or {}).get("STATUS") or "").strip()
        in USABLE_STATUSES
    }
    by_object_id = {
        centre.get("nea_object_id"): centre
        for centre in centres
        if centre.get("nea_object_id") is not None
    }
    by_postcode: dict[str, list[dict[str, Any]]] = {}
    for centre in centres:
        for postcode in extract_postcodes(centre.get("address", "")):
            by_postcode.setdefault(postcode, []).append(centre)

    added_centres: list[str] = []
    for feature in features:
        properties = feature.get("properties", {}) or {}
        status = str(properties.get("STATUS") or "").strip()
        if status not in USABLE_STATUSES:
            continue
        centre = by_object_id.get(properties.get("OBJECTID"))
        if centre is not None:
            _apply_feature(centre, feature)
            continue
        postal_code = str(properties.get("ADDRESSPOSTALCODE") or "").strip()
        existing_postcode_centres = by_postcode.get(postal_code, [])
        if len(existing_postcode_centres) == 1:
            _apply_feature(existing_postcode_centres[0], feature)
            continue
        new_centre = _centre_from_feature(feature)
        if new_centre["id"] in {item["id"] for item in centres}:
            new_centre["id"] = f"{new_centre['id']}_{properties.get('OBJECTID')}"
        centres.append(new_centre)
        added_centres.append(new_centre["name"])

    linked, unmatched = _match_list_rows(centres, rows)
    for row in rows:
        centre = linked.get(id(row))
        if centre is None:
            continue
        official_name = (row.get("name") or "").strip()
        previous = centre.get("official_name")
        if official_name and official_name != previous:
            aliases = list(centre.get("aliases", []) or [])
            for candidate in (previous, centre.get("name")):
                if candidate and candidate not in aliases and candidate != official_name:
                    aliases.append(candidate)
            centre["aliases"] = aliases
            centre["official_name"] = official_name
        centre["list_address"] = (row.get("address") or None)
        centre["managing_entity"] = (row.get("managing_entity") or None)
        centre["list_serial"] = row.get("serial")
        centre["list_table"] = row.get("table")
        centre["source_as_of"] = NEA_LIST_AS_OF

    listed_ids = {id(centre) for centre in linked.values()}
    retired: list[str] = []
    for centre in centres:
        # Every centre states where it came from, whichever source contributed it.
        centre["source_name"] = NEA_SOURCE_NAME
        centre["source_url"] = NEA_GEOJSON_DATASET_URL
        centre["source_as_of"] = NEA_LIST_AS_OF
        if id(centre) in listed_ids:
            centre["retired"] = False
            continue
        # A centre that the NEA geospatial data no longer lists as usable is
        # retired rather than silently kept in the catalogue.
        if centre.get("status") in USABLE_STATUSES and centre.get("nea_object_id") not in usable_object_ids:
            centre["retired"] = True
            retired.append(centre["name"])
        else:
            centre["retired"] = False

    return {
        "centres": centres,
        "added_centres": added_centres,
        "retired_centres": retired,
        "unmatched_list_rows": unmatched,
    }


def main() -> None:
    features = download_geojson()
    rows = download_centre_list()
    existing = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    result = build_catalogue(features, rows, existing)
    centres = result["centres"]

    if not centres:
        print("The refresh produced no hawker centres.")
        sys.exit(1)

    CENTRES_PATH.write_text(
        json.dumps(centres, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    meta = {
        "dataset_id": NEA_GEOJSON_DATASET_ID,
        "dataset_url": NEA_GEOJSON_DATASET_URL,
        "licence": NEA_LICENCE,
        "list_url": NEA_LIST_URL,
        "list_as_of": NEA_LIST_AS_OF,
        "geojson_record_updated": next(
            (
                (feature.get("properties", {}) or {}).get("FMEL_UPD_D")
                for feature in features
                if (feature.get("properties", {}) or {}).get("FMEL_UPD_D")
            ),
            None,
        ),
        "centre_count": len(centres),
        "list_row_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    CENTRES_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {len(centres)} hawker centres to {CENTRES_PATH.name}")
    if result["added_centres"]:
        print(f"  added: {', '.join(result['added_centres'])}")
    if result["retired_centres"]:
        print(f"  retired: {', '.join(result['retired_centres'])}")
    if result["unmatched_list_rows"]:
        print(f"  {len(result['unmatched_list_rows'])} official-list row(s) had no catalogue match:")
        for row in result["unmatched_list_rows"][:10]:
            print(f"    {row.get('serial')}: {row.get('name')} ({row.get('address')})")
    with_source = sum(1 for centre in centres if centre.get("source_as_of"))
    with_entity = sum(1 for centre in centres if centre.get("managing_entity"))
    print(f"  centres with source_as_of: {with_source}/{len(centres)}")
    print(f"  centres with a managing entity: {with_entity}/{len(centres)}")


if __name__ == "__main__":
    main()
