"""Optional community-reported data for stalls.

Some stalls are reported by the community rather than by an official source.
That data may fill a gap, but it may never stand in for a catalogue record
and it may never be presented as verified. Every value it supplies is named
``crowdsourced_*`` and carries the name of the community source, so a reply
can label it as reported rather than confirmed.

The whole feature is off unless ``ENABLE_CROWDSOURCED_SOURCE=1`` is set, and
a missing snapshot simply means no crowdsourced values at all.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from services.source_matching import (
    CentreMatcher, distinctive_tokens, normalise_name,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT_PATH = DATA_DIR / "hawkercentral_stalls.json"

SOURCE_NAME = "hawkercentral community listing"
SOURCE_KEY = "hawkercentral"
LICENCE = "MIT"

PRICED_STATUSES = frozenset({"fixed", "options", "variable"})

# A community listing older than this is reported as stale rather than as a
# current fact.
STALE_AFTER_DAYS = 90


def is_enabled() -> bool:
    """Return True when the community source is switched on."""
    return os.getenv("ENABLE_CROWDSOURCED_SOURCE", "").strip() in {
        "1", "true", "yes", "on",
    }


@lru_cache(maxsize=1)
def _snapshot() -> Optional[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _listing_date(entry: dict[str, Any], snapshot_as_of: Optional[str]) -> Optional[str]:
    """Return the date the community listing itself was last changed."""
    raw = str(entry.get("source_last_modified") or "").strip()
    if raw:
        return raw[:10]
    return snapshot_as_of


def _is_stale(as_of: Optional[str]) -> bool:
    """Return True when the listing is older than the freshness window."""
    if not as_of:
        return False
    try:
        listed = date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return False
    today = datetime.now(timezone.utc).date()
    return (today - listed).days > STALE_AFTER_DAYS


def get_crowdsourced_records() -> list[dict[str, Any]]:
    """Return the community records, or none when the source is switched off."""
    if not is_enabled():
        return []
    snapshot = _snapshot()
    if not snapshot:
        return []
    snapshot_as_of = snapshot.get("as_of")
    records = []
    for entry in snapshot.get("stalls", []):
        as_of = _listing_date(entry, snapshot_as_of)
        records.append({
            "name": str(entry.get("name") or "").strip(),
            "hawker_centre": str(entry.get("hawker_centre") or "").strip(),
            "centre_id": str(entry.get("centre_id") or "").strip() or None,
            "centre_name": str(entry.get("centre_name") or "").strip() or None,
            "unit_number": str(entry.get("unit") or entry.get("unit_number") or "").strip(),
            "postal": str(entry.get("postal") or "").strip(),
            "menu": entry.get("menu"),
            "price": entry.get("price"),
            "cuisine": entry.get("cuisine"),
            "opening": entry.get("opening"),
            "as_of": as_of,
            "is_stale": _is_stale(as_of),
        })
    return records


def link_to_stalls(
    records: list[dict[str, Any]],
    stalls: list[dict[str, Any]],
    centres: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Link the community records to the catalogue stalls."""
    matcher = CentreMatcher(centres)
    by_centre: dict[str, list[dict[str, Any]]] = {}
    for stall in stalls:
        centre_id = stall.get("hawker_centre_id")
        if centre_id:
            by_centre.setdefault(centre_id, []).append(stall)

    linked: dict[str, dict[str, Any]] = {}
    for record in records:
        centre, _method = matcher.match(
            centre_name=record.get("hawker_centre"),
        )
        centre_id = record.get("centre_id") or (centre or {}).get("id")
        if not centre_id:
            continue
        candidates = by_centre.get(centre_id, [])
        if not candidates:
            continue
        wanted = normalise_name(record.get("name"))
        if not wanted:
            continue
        unit = normalise_name(record.get("unit_number"))
        hits = [
            stall for stall in candidates
            if normalise_name(stall.get("name")) == wanted
            or (
                distinctive_tokens(record.get("name"))
                and distinctive_tokens(record.get("name"))
                <= distinctive_tokens(stall.get("name"))
            )
        ]
        if unit:
            same_unit = [
                stall for stall in hits
                if normalise_name(stall.get("location")) == unit
                or unit in normalise_name(stall.get("location"))
            ]
            if same_unit:
                hits = same_unit
        if len(hits) != 1:
            continue
        linked[hits[0]["stall_id"]] = record
    return linked


def unmatched_records(
    records: list[dict[str, Any]],
    stalls: list[dict[str, Any]],
    centres: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the community records that name no catalogue stall.

    A community record that sits at a NEA centre but matches no catalogue
    stall still describes a stall. It is kept as its own record, marked as
    community reported, so the gap in the catalogue is visible.
    """
    linked = link_to_stalls(records, stalls, centres)
    linked_records = {
        (record.get("centre_id") or "", normalise_name(record.get("name")))
        for record in linked.values()
    }
    unmatched = []
    for record in records:
        key = (record.get("centre_id") or "", normalise_name(record.get("name")))
        if key in linked_records:
            continue
        if record.get("centre_id"):
            unmatched.append(record)
    return unmatched


def apply_to_stall(stall: dict[str, Any], entry: Optional[dict[str, Any]]) -> None:
    """Add the community values that the stall record does not already have.

    A catalogue menu, a catalogue price, a cuisine or an opening time that is
    already recorded is never replaced by a community report.
    """
    for field in (
        "crowdsourced_menu",
        "crowdsourced_price",
        "crowdsourced_cuisine",
        "crowdsourced_opening",
        "crowdsourced_source",
        "crowdsourced_as_of",
        "crowdsourced_notice",
    ):
        stall.pop(field, None)
    if not entry:
        return

    menu_items = stall.get("menu_items") or []
    has_menu = bool(menu_items)
    has_price = any(
        item.get("price_status") in PRICED_STATUSES for item in menu_items
    )
    has_cuisine = bool(stall.get("cuisine"))
    has_opening = bool(stall.get("opening_hours"))

    if not has_menu and entry.get("menu"):
        stall["crowdsourced_menu"] = entry["menu"]
    if not has_price and entry.get("price"):
        stall["crowdsourced_price"] = entry["price"]
    if not has_cuisine and entry.get("cuisine"):
        stall["crowdsourced_cuisine"] = entry["cuisine"]
    if not has_opening and entry.get("opening"):
        stall["crowdsourced_opening"] = entry["opening"]
    stall["crowdsourced_source"] = SOURCE_NAME
    stall["crowdsourced_as_of"] = entry.get("as_of")
    if entry.get("is_stale"):
        stall["crowdsourced_notice"] = (
            "This community listing was last changed on "
            f"{entry.get('as_of')} and may no longer be current."
        )
