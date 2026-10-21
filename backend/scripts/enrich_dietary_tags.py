"""Enrich stall data with the HPB Healthier Dining Programme endorsement.

The Health Promotion Board publishes the partners of its Healthier Dining
Programme on data.gov.sg. The current dataset is a CSV of every endorsed
F&B outlet. The older GeoJSON dataset is still published and is used as a
fallback when the CSV cannot be downloaded.

Only the hawker settings of the dataset are used, because they are the
outlets a hawker-centre visitor can walk up to. An outlet is linked to a
catalogue stall when it names the same centre and the same trader: the
unit number when the dataset states one, and the trader name otherwise.

The script writes the source-owned fields ``healthier_dining``,
``healthier_criteria``, ``healthier_source`` and ``healthier_as_of`` onto the
matched stalls. It fails loudly when nothing matched, so a broken pipeline
can never look like a clean one.

Usage (from backend/):
    python scripts/enrich_dietary_tags.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.source_matching import (  # noqa: E402
    CentreMatcher, distinctive_tokens, extract_postcode, normalise_name,
    normalise_unit,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"
SNAPSHOT_PATH = DATA_DIR / "hpb_healthier_dining.json"

# The current dataset: the endorsed F&B partners, as a CSV.
HPB_DATASET_ID = "d_c475b9170c08f90580f4041bffd51639"
HPB_DATASET_URL = f"https://data.gov.sg/datasets/{HPB_DATASET_ID}/view"
HPB_DOWNLOAD_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    + HPB_DATASET_ID
    + "/poll-download"
)

# The older dataset of the same programme, kept as a fallback.
HPB_GEOJSON_DATASET_ID = "d_2925c2ccf75d1c135c2d469e0de3cee6"
HPB_GEOJSON_DOWNLOAD_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    + HPB_GEOJSON_DATASET_ID
    + "/poll-download"
)

HPB_LICENCE = "Singapore Open Data Licence (data.gov.sg)"
MANAGED_TAG = "healthier_dining_partner"
HPB_SOURCE_NAME = "HPB Healthier Dining Partners (F&B)"

HTTP_TIMEOUT_SECONDS = 60

CSV_COLUMNS = {
    "brand_name": "Brand_Name",
    "outlet_name": "Name_of_Outlet",
    "address": "Outlet_Address",
    "setting": "FnB_Setting",
    "criteria": "Menu_Item_Endorsement_Criteria",
    "region": "Regions",
    "company": "Company_Name",
}


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _poll_download(dataset_id: str) -> bytes:
    payload = requests.get(
        f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}"
        "/poll-download",
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    payload.raise_for_status()
    body = payload.json()
    if body.get("code") != 0:
        raise RuntimeError(f"HPB download failed: {body.get('errMsg')}")
    return _download(body["data"]["url"])


def _csv_bytes(payload: bytes) -> str:
    """Return the CSV text of a download, unpacking a zip when needed."""
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise RuntimeError("The HPB download held no CSV file.")
            payload = archive.read(names[0])
    return payload.decode("utf-8-sig", errors="replace")


def download_csv_outlets() -> list[dict[str, Any]]:
    """Download the current HPB Healthier Dining CSV and return its rows."""
    print(f"Downloading the HPB Healthier Dining dataset {HPB_DATASET_ID}...")
    text = _csv_bytes(_poll_download(HPB_DATASET_ID))
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise RuntimeError("The HPB dataset held no rows.")
    missing = [column for column in CSV_COLUMNS.values() if column not in rows[0]]
    if missing:
        raise RuntimeError(f"The HPB dataset is missing the columns: {', '.join(missing)}")
    outlets = []
    for row in rows:
        outlets.append({
            "brand_name": (row.get(CSV_COLUMNS["brand_name"]) or "").strip(),
            "outlet_name": (row.get(CSV_COLUMNS["outlet_name"]) or "").strip(),
            "address": (row.get(CSV_COLUMNS["address"]) or "").strip(),
            "setting": (row.get(CSV_COLUMNS["setting"]) or "").strip(),
            "criteria": (row.get(CSV_COLUMNS["criteria"]) or "").strip(),
            "region": (row.get(CSV_COLUMNS["region"]) or "").strip(),
            "company": (row.get(CSV_COLUMNS["company"]) or "").strip(),
        })
    print(f"  {len(outlets)} endorsed outlets")
    return outlets


def download_geojson_outlets() -> list[dict[str, Any]]:
    """Download the older HPB Healthier Eateries GeoJSON and return its rows."""
    print(f"Downloading the fallback HPB dataset {HPB_GEOJSON_DATASET_ID}...")
    payload = _poll_download(HPB_GEOJSON_DATASET_ID)
    features = json.loads(payload.decode("utf-8", errors="replace")).get("features", [])
    outlets = []
    for feature in features:
        properties = feature.get("properties", {}) or {}
        floor = str(properties.get("ADDRESSFLOORNUMBER") or "").strip()
        unit = str(properties.get("ADDRESSUNITNUMBER") or "").strip()
        address = str(properties.get("ADDRESS_MYENV") or properties.get("ADDRESS") or "")
        if floor and unit:
            address = f"{address} #{floor}-{unit}"
        outlets.append({
            "brand_name": str(properties.get("NAME") or "").strip(),
            "outlet_name": str(properties.get("NAME") or "").strip(),
            "address": address.strip(),
            "setting": "Hawker",
            "criteria": str(properties.get("HEALTHIER_DINING_CRITERIA") or "").strip(),
            "region": "",
            "company": "",
        })
    print(f"  {len(outlets)} endorsed outlets")
    return outlets


def _unit_of(value: Optional[str]) -> Optional[str]:
    """Return the comparable unit forms of an address."""
    if not value:
        return None
    match = re.search(r"#\s*([A-Za-z0-9/-]+)", str(value))
    if not match:
        return None
    forms = normalise_unit(match.group(1))
    return forms.pop() if forms else None


def _name_is_prefix(brand: str, name: str) -> bool:
    """Return True when the endorsed brand names the catalogue stall.

    The dataset sometimes records the trader name and the catalogue records
    the trader together with what it sells, so the brand has to be the start
    of the catalogue name.
    """
    wanted = normalise_name(brand)
    candidate = normalise_name(name)
    if not wanted or not candidate:
        return False
    return candidate == wanted or candidate.startswith(wanted + " ")


def link_outlets(
    outlets: list[dict[str, Any]],
    stalls: list[dict[str, Any]],
    centres: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Link the endorsed outlets to the catalogue stalls."""
    matcher = CentreMatcher(centres)
    by_centre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stall in stalls:
        centre_id = stall.get("hawker_centre_id")
        if centre_id:
            by_centre[centre_id].append(stall)

    linked: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = defaultdict(int)
    for outlet in outlets:
        if outlet["setting"].strip().casefold() != "hawker":
            continue
        stats["hawker_outlets"] += 1
        postcode = extract_postcode(outlet["address"])
        centre, method = matcher.match(
            postcode=postcode,
            centre_name=outlet["outlet_name"],
            address=outlet["address"],
        )
        if not centre:
            stats["without_centre"] += 1
            continue
        candidates = by_centre.get(centre["id"], [])
        if not candidates:
            stats["centre_without_catalogue"] += 1
            continue
        stats["centre_with_catalogue"] += 1

        hits = [
            stall for stall in candidates
            if distinctive_tokens(outlet["brand_name"])
            and _name_is_prefix(outlet["brand_name"], stall.get("name", ""))
        ]
        if len(hits) != 1:
            unit = _unit_of(outlet["address"])
            unit_hits = [
                stall for stall in candidates
                if unit and unit in normalise_unit(stall.get("location"))
            ]
            hits = unit_hits if len(unit_hits) == 1 else []
        if len(hits) != 1:
            stats["ambiguous" if len(hits) > 1 else "unmatched"] += 1
            continue
        stall = hits[0]
        stats["matched"] += 1
        existing = linked.get(stall["stall_id"])
        if existing:
            existing["healthier_criteria"] = _merge_criteria(
                existing["healthier_criteria"], outlet["criteria"]
            )
            continue
        linked[stall["stall_id"]] = {
            "healthier_criteria": outlet["criteria"],
            "healthier_source": HPB_SOURCE_NAME,
            "healthier_match": "trader_name" if _name_is_prefix(
                outlet["brand_name"], stall.get("name", "")
            ) else "unit_number",
            "healthier_brand_name": outlet["brand_name"],
        }
    return linked, dict(stats)


def _merge_criteria(current: str, addition: str) -> str:
    """Merge two endorsement criteria strings without repeating a criterion."""
    parts = [part.strip() for part in current.split(",") if part.strip()]
    for part in (item.strip() for item in addition.split(",")):
        if part and part.casefold() not in {item.casefold() for item in parts}:
            parts.append(part)
    return ", ".join(parts)


def enrich(stalls_path: Path) -> int:
    """Write the endorsement onto the matched stalls of one file."""
    if not stalls_path.exists():
        return 0
    payload = json.loads(stalls_path.read_text(encoding="utf-8"))
    stalls = payload.get("stalls", [])
    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    outlets = _OUTLETS
    linked, stats = link_outlets(outlets, stalls, centres)
    as_of = _AS_OF
    tagged = 0
    for stall in stalls:
        match = linked.get(stall.get("stall_id"))
        if not match:
            stall.pop("healthier_dining", None)
            stall.pop("healthier_criteria", None)
            stall.pop("healthier_source", None)
            stall.pop("healthier_as_of", None)
            continue
        stall["healthier_dining"] = True
        stall["healthier_criteria"] = match["healthier_criteria"]
        stall["healthier_source"] = match["healthier_source"]
        stall["healthier_as_of"] = as_of
        tags = set(stall.get("dietary_tags", []))
        tags.add(MANAGED_TAG)
        stall["dietary_tags"] = sorted(tags)
        tagged += 1
    if tagged:
        payload["stalls"] = stalls
        stalls_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return tagged


_OUTLETS: list[dict[str, Any]] = []
_AS_OF: Optional[str] = None


def main() -> None:
    global _OUTLETS, _AS_OF
    try:
        outlets = download_csv_outlets()
        dataset_id = HPB_DATASET_ID
        dataset_url = HPB_DATASET_URL
    except Exception as error:  # noqa: BLE001 - report the failure and fall back
        print(f"  the current dataset is unavailable: {error}")
        outlets = download_geojson_outlets()
        dataset_id = HPB_GEOJSON_DATASET_ID
        dataset_url = (
            f"https://data.gov.sg/datasets/{HPB_GEOJSON_DATASET_ID}/view"
        )

    _OUTLETS = outlets
    _AS_OF = datetime.now(timezone.utc).date().isoformat()

    stalls = json.loads(STALLS_PATH.read_text(encoding="utf-8")).get("stalls", [])
    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    linked, stats = link_outlets(outlets, stalls, centres)
    print("\nLinking the endorsed outlets to the catalogue:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  catalogue stalls matched: {len(linked)} of {len(stalls)}")

    SNAPSHOT_PATH.write_text(
        json.dumps({
            "source": HPB_SOURCE_NAME,
            "dataset_id": dataset_id,
            "dataset_url": dataset_url,
            "licence": HPB_LICENCE,
            "as_of": _AS_OF,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "outlet_count": len(outlets),
            "linked_stall_count": len(linked),
            "link_stats": stats,
            "outlets": [
                {
                    "brand_name": outlet["brand_name"],
                    "outlet_name": outlet["outlet_name"],
                    "address": outlet["address"],
                    "setting": outlet["setting"],
                    "criteria": outlet["criteria"],
                    "region": outlet["region"],
                }
                for outlet in outlets
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tagged = enrich(STALLS_PATH)
    print(f"\nTagged {tagged} catalogue stalls in {STALLS_PATH.name}")
    if tagged == 0:
        print(
            "No stalls were tagged. The HPB data may use different naming or "
            "unit conventions than the catalogue. The pipeline is left without "
            "a Healthier Dining endorsement rather than with a wrong one."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
