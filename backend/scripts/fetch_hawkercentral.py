"""Snapshot the community hawker listing.

The hawkercentral project collects hawker and stall information from the
community. The snapshot is optional: the script only runs when
``ENABLE_CROWDSOURCED_SOURCE=1`` is set, and the source address is
configurable through ``HAWKERCENTRAL_SOURCE_URL``.

The default source is the public, unauthenticated Firestore REST endpoint the
project publishes. No page is scraped and no private endpoint is used.

Only records that belong to a NEA hawker centre are kept. A record whose
postal code does not match a NEA centre exactly is dropped, and so is a record
that is clearly not a hawker stall, such as a grocer or a delivery-only
outlet. The values that are kept may only fill a gap in the catalogue, never
replace it, and a reply must label them as community reported.

Usage (from backend/):
    ENABLE_CROWDSOURCED_SOURCE=1 python scripts/fetch_hawkercentral.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.crowdsourced_service import (  # noqa: E402
    LICENCE, SOURCE_NAME, SNAPSHOT_PATH,
)

DEFAULT_SOURCE_URL = os.getenv(
    "HAWKERCENTRAL_SOURCE_URL",
    "https://firestore.googleapis.com/v1/projects/hawkercentral/databases/"
    "(default)/documents/hawkers",
)
HTTP_TIMEOUT_SECONDS = float(os.getenv("HAWKERCENTRAL_TIMEOUT_SECONDS", "60"))
PAGE_SIZE = 200
MAX_PAGES = 100

# The categories that never describe a hawker centre stall. A grocer, a
# restaurant or a delivery-only outlet listed on the community site is not a
# hawker stall, so its record is dropped rather than shown as one.
NON_HAWKER_CATEGORY_WORDS = (
    "grocery",
    "supermarket",
    "mini mart",
    "minimart",
    "convenience store",
    "wet market",
    "restaurant",
    "food court",
    "foodcourt",
    "bar ",
    "pub",
    "wine",
    "beer",
    "clothing",
    "pharmacy",
    "salon",
    "hardware",
    "florist",
    "bakery",
    "deli",
    "catering service",
)

# A record that only offers delivery across the island is not a stall in a
# hawker centre, whatever its category says.
DELIVERY_ONLY_VALUES = {"islandwide", "island wide", "singapore"}


def _decode_value(value: dict[str, Any]) -> Any:
    """Return the plain value of a Firestore field."""
    if "stringValue" in value:
        return value["stringValue"]
    if "booleanValue" in value:
        return value["booleanValue"]
    if "integerValue" in value:
        try:
            return int(value["integerValue"])
        except (TypeError, ValueError):
            return value["integerValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "timestampValue" in value:
        return value["timestampValue"]
    if "nullValue" in value:
        return None
    if "arrayValue" in value:
        return [
            _decode_value(item)
            for item in value["arrayValue"].get("values", [])
        ]
    if "mapValue" in value:
        return {
            key: _decode_value(item)
            for key, item in value["mapValue"].get("fields", {}).items()
        }
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value if _text(item))
    if isinstance(value, dict):
        label = value.get("label") or value.get("value")
        return _text(label)
    return str(value).strip()


def _menu_entries(raw: Any) -> list[dict[str, Any]]:
    """Return the menu of a record in the shape the catalogue uses."""
    entries = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"))
        if not name:
            continue
        entries.append({
            "name": name,
            "price": _text(item.get("price")) or None,
            "description": _text(item.get("description")) or None,
            # An absent value means the source did not say, not that the dish
            # is unavailable.
            "available": (
                bool(item["available"]) if "available" in item else None
            ),
            "variants": item.get("variants") or [],
            "section": _text(item.get("section")) or None,
            "image": _text(item.get("image")) or None,
        })
    return entries


def to_record(document: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the flat record of one source document."""
    fields = document.get("fields") or {}
    values = {key: _decode_value(value) for key, value in fields.items()}
    name = _text(values.get("name"))
    if not name:
        return None
    location = values.get("location") or {}
    geo = location.get("geopoint") if isinstance(location, dict) else None
    latitude = values.get("latitude")
    longitude = values.get("longitude")
    if geo and isinstance(geo, dict):
        latitude = geo.get("latitude")
        longitude = geo.get("longitude")
    return {
        "name": name,
        "hawker_centre": _text(values.get("hawker_centre")),
        "postal": _text(values.get("postal")).replace(" ", ""),
        "unit": _text(values.get("unit")),
        "street": _text(values.get("street")),
        "location": {
            "lat": _text(latitude) or None,
            "lng": _text(longitude) or None,
        },
        "cuisine": _text(values.get("cuisine")),
        "categories": _text(values.get("categories")),
        "tags": _text(values.get("tagsValue") or values.get("tags")),
        "opening": _text(values.get("opening")),
        "contact": _text(values.get("contact")),
        "menu": _menu_entries(values.get("menu_combined")),
        "price": _text(values.get("price")) or None,
        "source_doc_id": (document.get("name") or "").rsplit("/", 1)[-1],
        "source_last_modified": _text(values.get("lastmodified")),
        "inperson": values.get("inperson"),
        "delivery_option": values.get("delivery_option"),
        "regions": _text(values.get("regions") or values.get("region")),
        "custom": values.get("custom"),
    }


def download_documents(url: str) -> list[dict[str, Any]]:
    """Download every record of the community collection."""
    print(f"Downloading the community listing from {url}...")
    documents: list[dict[str, Any]] = []
    page_token = None
    for _page in range(MAX_PAGES):
        params = {"pageSize": PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
        documents.extend(body.get("documents") or [])
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    print(f"  {len(documents)} community documents")
    return documents


def _category_text(record: dict[str, Any]) -> str:
    return " ".join(
        _text(record.get(field)).casefold()
        for field in ("categories", "cuisine", "tags")
    )


def _is_delivery_only(record: dict[str, Any]) -> bool:
    """Return True when the record offers delivery only."""
    regions = _text(record.get("regions")).casefold()
    if regions and all(
        value in DELIVERY_ONLY_VALUES
        for value in [part.strip() for part in regions.split(",") if part.strip()]
    ):
        return True
    if record.get("inperson") is False and (
        record.get("delivery_option") or record.get("regions")
    ):
        return True
    return False


def filter_records(
    records: list[dict[str, Any]],
    centre_postcodes: set[str],
    centres_by_postcode: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the records that belong to a NEA hawker centre.

    A record is kept when its postal code matches a NEA centre exactly and it
    is not clearly something other than a hawker stall. Every dropped record
    is counted so the gap is visible.
    """
    kept: list[dict[str, Any]] = []
    dropped = {
        "no_postal": 0,
        "postal_not_a_nea_centre": 0,
        "not_a_hawker_stall": 0,
        "delivery_only": 0,
    }
    for record in records:
        postal = str(record.get("postal") or "").strip()
        if not postal:
            dropped["no_postal"] += 1
            continue
        if postal not in centre_postcodes:
            dropped["postal_not_a_nea_centre"] += 1
            continue
        category_text = _category_text(record)
        if any(word in category_text for word in NON_HAWKER_CATEGORY_WORDS):
            dropped["not_a_hawker_stall"] += 1
            continue
        if _is_delivery_only(record):
            dropped["delivery_only"] += 1
            continue
        centre = (centres_by_postcode or {}).get(postal)
        if centre:
            record["centre_id"] = centre.get("id")
            record["centre_name"] = centre.get("name")
        kept.append(record)
    return kept, dropped


def load_centres_by_postcode() -> dict[str, dict[str, Any]]:
    """Return the NEA hawker centres of every postal code."""
    from services.location_service import load_hawker_centres

    centres: dict[str, dict[str, Any]] = {}
    for centre in load_hawker_centres():
        codes = set()
        if centre.get("postal_code"):
            codes.add(str(centre["postal_code"]).strip())
        for value in str(centre.get("address") or "").split():
            candidate = value.strip()
            if len(candidate) == 6 and candidate.isdigit():
                codes.add(candidate)
        for code in codes:
            centres.setdefault(code, centre)
    return centres


def load_centre_postcodes() -> set[str]:
    """Return the postal codes of the NEA hawker centres."""
    return set(load_centres_by_postcode())


def main() -> None:
    if os.getenv("ENABLE_CROWDSOURCED_SOURCE", "").strip() not in {
        "1", "true", "yes", "on",
    }:
        print(
            "The community source is off. Set ENABLE_CROWDSOURCED_SOURCE=1 to "
            "snapshot it."
        )
        sys.exit(2)

    url = os.getenv("HAWKERCENTRAL_SOURCE_URL", DEFAULT_SOURCE_URL)
    try:
        documents = download_documents(url)
    except (requests.RequestException, ValueError) as error:
        print(f"  the community listing could not be read: {error}")
        print("  the snapshot was left unchanged.")
        sys.exit(1)
    if not documents:
        print("  the community listing held no records. The snapshot was left unchanged.")
        sys.exit(1)

    records = [record for record in (to_record(doc) for doc in documents) if record]
    print(f"  {len(records)} community records with a name")

    kept, dropped = filter_records(
        records,
        load_centre_postcodes(),
        load_centres_by_postcode(),
    )
    print(f"  {len(kept)} records are at a NEA hawker centre")
    for reason, count in dropped.items():
        print(f"    dropped {count}: {reason.replace('_', ' ')}")

    with_menu = [record for record in kept if record.get("menu")]
    with_price = [
        record for record in kept
        if record.get("price") or any(item.get("price") for item in record.get("menu") or [])
    ]
    print(f"  {len(with_menu)} records hold a menu")
    print(f"  {len(with_price)} records hold a menu or a price")

    if not kept:
        print("  no record matched a NEA hawker centre. The snapshot was left unchanged.")
        sys.exit(1)

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    SNAPSHOT_PATH.write_text(
        json.dumps({
            "source": SOURCE_NAME,
            "source_url": url,
            "licence": LICENCE,
            "as_of": retrieved_at[:10],
            "retrieved_at": retrieved_at,
            "record_count": len(kept),
            "downloaded_count": len(records),
            "dropped": dropped,
            "menu_record_count": len(with_menu),
            "priced_record_count": len(with_price),
            "stalls": kept,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SNAPSHOT_PATH.name}")


if __name__ == "__main__":
    main()
