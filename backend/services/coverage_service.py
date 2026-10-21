"""Per-centre coverage of what the catalogue actually holds.

The coverage report lets the assistant state a true fact instead of
hedging: how many cooked-food stalls NEA records at a centre, how many of
them the catalogue has menu data for, and which sources contribute to that
centre at all. Real data, synthetic demo data and crowdsourced data are
counted separately, because they may never be presented as the same thing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_FILE = DATA_DIR / "hawker_centres.json"
STALLS_FILE = DATA_DIR / "hawker_stall.json"

PRICED_STATUSES = frozenset({"fixed", "options", "variable"})


def _read(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def _centres() -> tuple[dict[str, Any], ...]:
    centres = _read(CENTRES_FILE) or []
    return tuple(centres if isinstance(centres, list) else [])


@lru_cache(maxsize=1)
def _stalls_by_centre() -> dict[str, list[dict[str, Any]]]:
    """Return the catalogue stalls of every centre."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    payload = _read(STALLS_FILE) or {}
    for stall in payload.get("stalls", []):
        centre_id = stall.get("hawker_centre_id")
        if not centre_id:
            continue
        grouped.setdefault(centre_id, []).append(stall)
    # The community source is optional and holds stalls that the catalogue
    # does not. They are counted in their own bucket.
    try:
        from services.hawker_service import _crowdsourced_stalls

        centres = list(_centres())
        stalls = list(payload.get("stalls", []))
        for stall in _crowdsourced_stalls(stalls, centres):
            centre_id = stall.get("hawker_centre_id")
            if not centre_id:
                continue
            grouped.setdefault(centre_id, []).append(stall)
    except Exception:  # pragma: no cover - the community source is optional
        pass
    return grouped


@lru_cache(maxsize=1)
def _coverage() -> dict[str, dict[str, Any]]:
    """Return the coverage of every centre."""
    grouped = _stalls_by_centre()
    coverage: dict[str, dict[str, Any]] = {}
    for centre in _centres():
        centre_id = centre.get("id")
        if not centre_id:
            continue
        stalls = grouped.get(centre_id, [])
        real = [stall for stall in stalls if not stall.get("is_mock")]
        synthetic = [stall for stall in stalls if stall.get("is_mock")]
        crowdsourced = [
            stall for stall in stalls
            if stall.get("data_origin") == "crowdsourced_unverified"
        ]
        menu_items = [
            item
            for stall in real
            for item in stall.get("menu_items", [])
        ]
        sources = ["nea_hawker_centres"]
        if real:
            sources.append("wak_wak_hawker")
        if synthetic:
            sources.append("synthetic_demo")
        if crowdsourced:
            sources.append("hawkercentral")
        if centre.get("merchant_stalls"):
            sources.append("dbs_paylah_directory")
        if any(stall.get("healthier_dining") for stall in stalls):
            sources.append("hpb_healthier_dining")
        if any(stall.get("halal_certified") for stall in stalls):
            sources.append("muis_halal")
        merchants = centre.get("merchant_stalls") or []
        coverage[centre_id] = {
            "nea_cooked_food_stalls": centre.get("cooked_food_stalls"),
            "catalogue_stalls": len(real),
            "menu_items": len(menu_items),
            "priced_items": sum(
                1 for item in menu_items
                if item.get("price_status") in PRICED_STATUSES
            ),
            "halal_certified_stalls": sum(
                1 for stall in stalls if stall.get("halal_certified") is True
            ),
            "healthier_dining_stalls": sum(
                1 for stall in stalls if stall.get("healthier_dining") is True
            ),
            "merchant_listed_stalls": len(merchants),
            # The directory lists some units without a merchant name. They are
            # counted, but they must never be presented as a named stall.
            "merchant_entries_without_name": sum(
                1 for merchant in merchants if not merchant.get("name")
            ),
            "crowdsourced_stalls": len(crowdsourced),
            "synthetic_stalls": len(synthetic),
            "sources": sources,
        }
    return coverage


def get_coverage(centre_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Return the coverage of one centre, or None when it is unknown."""
    if not centre_id:
        return None
    entry = _coverage().get(centre_id)
    return dict(entry) if entry else None


def coverage_summary() -> dict[str, Any]:
    """Return the coverage of the whole catalogue."""
    entries = _coverage().values()
    return {
        "centres": len(entries),
        "centres_with_menu_items": sum(1 for entry in entries if entry["menu_items"]),
        "catalogue_stalls": sum(entry["catalogue_stalls"] for entry in entries),
        "menu_items": sum(entry["menu_items"] for entry in entries),
        "priced_items": sum(entry["priced_items"] for entry in entries),
        "halal_certified_stalls": sum(
            entry["halal_certified_stalls"] for entry in entries
        ),
        "healthier_dining_stalls": sum(
            entry["healthier_dining_stalls"] for entry in entries
        ),
        "merchant_listed_stalls": sum(
            entry["merchant_listed_stalls"] for entry in entries
        ),
        "merchant_entries_without_name": sum(
            entry["merchant_entries_without_name"] for entry in entries
        ),
        "synthetic_stalls": sum(entry["synthetic_stalls"] for entry in entries),
        "crowdsourced_stalls": sum(
            entry["crowdsourced_stalls"] for entry in entries
        ),
    }
