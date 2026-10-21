"""Read and locally filter the scraped stall and menu catalogue."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from services.location_service import load_hawker_centres


STALLS_FILE = Path(__file__).resolve().parents[1] / "data" / "hawker_stall.json"

# Dietary preferences that require structured data (ingredients, allergens,
# certification) to verify. The Wak-Wak catalogue does not have these fields,
# so any verifiable restriction cannot be confirmed from the source alone and
# must short-circuit to an empty result.
# Diets with structured certification/allergen data that can be verified:
#   vegetarian, vegan, no pork, no beef, gluten free, dairy free, nut free,
#   seafood free
VERIFIABLE_DIETS = frozenset({
    "vegetarian", "vegan", "no pork", "no beef",
    "gluten free", "dairy free", "nut free", "seafood free",
})


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _record_sources(stall: dict[str, Any]) -> list[str]:
    """Return the sources that hold a value for this stall."""
    if stall.get("data_origin") == "crowdsourced_unverified":
        return ["crowdsourced"]
    sources = ["wak_wak_hawker"]
    if stall.get("is_mock"):
        sources = ["synthetic_demo"]
    if stall.get("healthier_dining"):
        sources.append("hpb_healthier_dining")
    if stall.get("halal_certified"):
        sources.append("muis_halal")
    if stall.get("merchant_list_name"):
        sources.append("dbs_paylah_directory")
    if stall.get("crowdsourced_source"):
        sources.append("crowdsourced")
    return sources


def _crowdsourced_fields(
    stalls: list[dict[str, Any]], centres: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return the community values of every stall, where they fill a gap."""
    from services import crowdsourced_service

    records = crowdsourced_service.get_crowdsourced_records()
    if not records:
        return {}
    linked = crowdsourced_service.link_to_stalls(records, stalls, centres)
    fields: dict[str, dict[str, Any]] = {}
    for stall in stalls:
        stall_id = stall.get("stall_id")
        entry = linked.get(stall_id)
        if not entry:
            continue
        probe = {
            "menu_items": [dict(item) for item in stall.get("menu_items", [])],
            "cuisine": stall.get("category"),
            "opening_hours": stall.get("opening_hours"),
        }
        crowdsourced_service.apply_to_stall(probe, entry)
        filled = {
            key: value for key, value in probe.items()
            if key.startswith("crowdsourced_") and value is not None
        }
        if filled:
            fields[stall_id] = filled
    return fields


def _crowdsourced_stalls(
    stalls: list[dict[str, Any]], centres: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the community stalls that name no catalogue stall.

    A community record that sits at a NEA centre but matches no catalogue
    stall still describes a stall. It becomes its own record, marked as
    community reported, so the gap in the catalogue is visible rather than
    silently missing.
    """
    from services import crowdsourced_service

    records = crowdsourced_service.get_crowdsourced_records()
    if not records:
        return []
    by_id = {centre["id"]: centre for centre in centres}
    results = []
    for record in crowdsourced_service.unmatched_records(records, stalls, centres):
        centre = by_id.get(record.get("centre_id") or "")
        if not centre:
            continue
        name = record.get("name")
        if not name:
            continue
        menu = record.get("menu") or []
        if not menu:
            # A community record with no dish describes no dish to recommend.
            continue
        items = []
        for index, entry in enumerate(menu):
            item_name = str(entry.get("name") or "").strip()
            if not item_name:
                continue
            price_label = str(entry.get("price") or "").strip() or None
            items.append({
                "menu_item_id": f"crowdsourced_{centre['id']}_{index}",
                "name": item_name,
                "price_status": "fixed" if price_label else "unavailable",
                "price_sgd_min": _price_amount(price_label),
                "price_sgd_max": _price_amount(price_label),
                "dietary_tags": [],
            })
        if not items:
            continue
        stall_id = f"crowdsourced_{centre['id']}_{_normalise(name).replace(' ', '_')}"
        results.append({
            "stall_id": stall_id,
            "hawker_centre_id": centre["id"],
            "name": name,
            "location": record.get("unit_number") or "",
            "category": record.get("cuisine") or "",
            "menu_items": items,
            "crowdsourced_menu": menu,
            "crowdsourced_price": record.get("price"),
            "crowdsourced_cuisine": record.get("cuisine"),
            "crowdsourced_opening": record.get("opening"),
            "crowdsourced_source": "hawkercentral community listing",
            "crowdsourced_as_of": record.get("as_of"),
            "crowdsourced_is_stale": bool(record.get("is_stale")),
            "data_as_of": record.get("as_of"),
            "data_origin": "crowdsourced_unverified",
            "is_mock": False,
        })
    return results


def _price_amount(label: Optional[str]) -> Optional[float]:
    """Return the amount of a price label such as "$7.00"."""
    if not label:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(label))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _menu_catalogue() -> tuple[dict[str, Any], ...]:
    payload = json.loads(STALLS_FILE.read_text(encoding="utf-8"))
    centres = {item["id"]: item for item in load_hawker_centres()}
    stalls = payload.get("stalls", [])
    crowdsourced = _crowdsourced_fields(stalls, list(centres.values()))
    community_stalls = _crowdsourced_stalls(stalls, list(centres.values()))
    records = []
    for stall in stalls + community_stalls:
        centre_id = stall.get("hawker_centre_id")
        centre = centres.get(centre_id) if centre_id else None
        centre_match_method = stall.get("centre_match_method")
        # The location agent only recommends centres in the NEA catalogue.
        # Keep unmatched source records in the JSON, but don't present them as
        # verified hawker-centre options.
        if not centre:
            # Fall back to matching by hawker_centre text name against
            # catalogue centre names (e.g. Wak-Wak stalls with no
            # hawker_centre_id linking them to the NEA catalogue).
            wanted_name = _normalise(stall.get("hawker_centre") or "")
            if wanted_name:
                for c in centres.values():
                    if _normalise(c.get("name")) == wanted_name or _normalise(
                        c.get("official_name", "")
                    ) == wanted_name:
                        centre = c
                        centre_id = c["id"]
                        if centre_match_method is None:
                            centre_match_method = "name_unique"
                        break
            if not centre:
                continue
        categories = [
            part.strip()
            for part in str(stall.get("category") or "").split(",")
            if part.strip()
        ]
        unit = re.search(r"#\s*([A-Za-z0-9/-]+)", str(stall.get("location") or ""))
        # A community stall carries its own values; a catalogue stall carries
        # the values that filled a gap. Both end up on the record.
        community_values = {
            key: stall.get(key)
            for key in (
                "crowdsourced_menu",
                "crowdsourced_price",
                "crowdsourced_cuisine",
                "crowdsourced_opening",
                "crowdsourced_notice",
            )
            if stall.get(key) is not None
        }
        crowd_values = {
            **crowdsourced.get(stall.get("stall_id"), {}),
            **community_values,
        }
        for item in stall.get("menu_items", []):
            item_name = str(item.get("name") or "").strip()
            if not item_name:
                continue
            minimum = item.get("price_sgd_min")
            maximum = item.get("price_sgd_max")
            price_status = item.get("price_status", "unparsed")
            if price_status == "fixed" and minimum is not None:
                price_label = f"${minimum:.2f}"
            elif price_status == "options" and item.get("price_option_labels"):
                labels = item["price_option_labels"]
                options = item.get("price_sgd_options", [])
                price_label = ", ".join(
                    f"{label} ${amount:.2f}"
                    for label, amount in zip(labels, options)
                )
            elif price_status == "options":
                price_label = "Listed price options: " + " / ".join(
                    f"${amount:.2f}" for amount in item.get("price_sgd_options", [])
                )
            elif price_status == "variable":
                price_label = "Varies with selected ingredients"
            else:
                price_label = "not listed as a parseable price"
            records.append({
                "stall_id": stall.get("stall_id"),
                "hawker_centre_id": centre_id,
                "hawker_centre": centre["name"],
                "hawker_centre_source_name": stall.get("hawker_centre"),
                "stall_name": str(stall.get("name") or "").strip(),
                "unit_number": f"#{unit.group(1)}" if unit else None,
                "menu_item_id": item.get("menu_item_id"),
                "menu_item_name": item_name,
                "menu_item": item_name,
                "categories": categories,
                "cuisine": categories[0] if categories else None,
                "price": minimum,
                "price_currency": "SGD" if item.get("price_sgd_options") else None,
                "price_label": price_label,
                "price_sgd": item.get("price_sgd"),
                "price_sgd_min": item.get("price_sgd_min"),
                "price_sgd_max": item.get("price_sgd_max"),
                "price_sgd_options": item.get("price_sgd_options", []),
                "price_option_labels": item.get("price_option_labels", []),
                "price_model": item.get("price_model", "unknown"),
                "price_status": item.get("price_status", "unparsed"),
                "price_basis": item.get("price_basis", "unknown"),
                "price_raw": item.get("price_raw", item.get("price")),
                # Read dietary tags from source data (may be enriched from HPB
                # Healthier Dining or other external sources). Tags are added
                # at the stall level and inherited by each menu item.
                "dietary_tags": item.get("dietary_tags", stall.get("dietary_tags", [])),
                "dietary_suitable": item.get("dietary_suitable", None),
                "source_url": stall.get("url"),
                "data_origin": stall.get("data_origin", "wak_wak_source"),
                "is_mock": bool(stall.get("is_mock", False)),
                "mock_notice": stall.get("mock_notice"),
                # Source-owned fields. Each one states which source it came
                # from and the date that source is as of, so a reply can name
                # the source instead of asserting the fact on its own.
                "halal_certified": stall.get("halal_certified"),
                "halal_certificate_no": stall.get("halal_certificate_no"),
                "halal_source": stall.get("halal_source"),
                "halal_as_of": stall.get("halal_as_of"),
                "healthier_dining": stall.get("healthier_dining"),
                "healthier_criteria": stall.get("healthier_criteria"),
                "healthier_source": stall.get("healthier_source"),
                "healthier_as_of": stall.get("healthier_as_of"),
                "merchant_list_name": stall.get("merchant_list_name"),
                "merchant_unit_number": stall.get("merchant_unit_number"),
                "merchant_source": stall.get("merchant_source"),
                "centre_match_method": centre_match_method,
                "data_as_of": stall.get("data_as_of"),
                "sources": _record_sources(stall),
                **crowd_values,
            })
    return tuple(records)


def _filtered_menu(
    *,
    candidate_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    allowed_ids = {value for value in candidate_centre_ids or [] if value}
    if not allowed_ids and candidate_centres:
        wanted = {_normalise(name) for name in candidate_centres if name}
        allowed_ids = {
            centre["id"]
            for centre in load_hawker_centres()
            if _normalise(centre["name"]) in wanted
            or _normalise(centre.get("official_name", "")) in wanted
            or any(_normalise(alias) in wanted for alias in centre.get("aliases", []))
        }
    return [
        item for item in _menu_catalogue()
        if not allowed_ids or item["hawker_centre_id"] in allowed_ids
    ]


@lru_cache(maxsize=1)
def _menu_by_id() -> dict[str, dict[str, Any]]:
    return {
        item["menu_item_id"]: item
        for item in _menu_catalogue()
        if item.get("menu_item_id")
    }


def get_menu_item_record(menu_item_id: str) -> Optional[dict[str, Any]]:
    """Return source-backed menu data used by the verifier."""
    item = _menu_by_id().get(menu_item_id)
    return dict(item) if item else None


def get_menu_stall_id(stall_name: str, hawker_centre_id: Optional[str]) -> Optional[str]:
    """Resolve an exact queue-feed stall label to a catalogue stall ID."""
    wanted = _normalise(stall_name)
    if not wanted or not hawker_centre_id:
        return None
    for item in _menu_catalogue():
        if (
            item["hawker_centre_id"] == hawker_centre_id
            and _normalise(item["stall_name"]) == wanted
        ):
            return item["stall_id"]
    return None


def get_merchant_stalls(
    centre_ids: list[str],
    terms: Optional[list[str]] = None,
    limit_per_centre: int = 6,
) -> list[dict[str, Any]]:
    """Return the merchant-listed stalls of the given centres.

    The merchant directory lists real stall names and unit numbers, but no
    menu, price, dietary or opening data. It is the only source that names
    the stalls of a centre the menu catalogue does not cover, so a reply can
    name them and say what is not known.
    """
    wanted_ids = [value for value in (centre_ids or []) if value]
    if not wanted_ids:
        return []
    centres = {item["id"]: item for item in load_hawker_centres()}
    search_terms = [
        _normalise(term) for term in (terms or []) if str(term or "").strip()
    ]
    entries = []
    for centre_id in wanted_ids:
        centre = centres.get(centre_id)
        if not centre:
            continue
        merchants = centre.get("merchant_stalls") or []
        matched = [
            merchant for merchant in merchants
            if merchant.get("name")
            and (
                not search_terms
                or any(
                    term in _normalise(merchant.get("name"))
                    for term in search_terms
                )
            )
        ]
        if not matched:
            continue
        entries.append({
            "centre_id": centre_id,
            "centre_name": centre.get("name"),
            "merchants": [
                {
                    "name": merchant.get("name"),
                    "unit_number": merchant.get("unit_number"),
                }
                for merchant in matched[:limit_per_centre]
            ],
            "merchant_count": len(matched),
            "source": centre.get("merchant_source"),
            "as_of": centre.get("merchant_as_of"),
            "note": (
                "The merchant directory lists the stall name and unit only. It "
                "carries no menu, price, dietary or opening data for these "
                "stalls."
            ),
        })
    return entries


@lru_cache(maxsize=1)
def _stall_by_id() -> dict[str, dict[str, Any]]:
    return {
        item["stall_id"]: item
        for item in _menu_catalogue()
        if item.get("stall_id")
    }


def get_menu_stall_record(stall_id: str) -> Optional[dict[str, Any]]:
    item = _stall_by_id().get(stall_id)
    return dict(item) if item else None


def _matches_category(preference: str, categories: list[str]) -> bool:
    wanted = _normalise(preference)
    if not wanted:
        return False
    return any(
        wanted == _normalise(category)
        or wanted in _normalise(category).split()
        for category in categories
    )


def find_nearby_stalls(
    location: str = "",
    candidate_centre_ids: Optional[list[str]] = None,
):
    """Return menu-backed candidates in the centre shortlist supplied by Location."""
    return _filtered_menu(candidate_centre_ids=candidate_centre_ids)


def find_dietary_stalls(
    dietary_preferences: Optional[list[str]] = None,
    cuisine_preferences: Optional[list[str]] = None,
    menu_item_preferences: Optional[list[str]] = None,
    stall_name_preferences: Optional[list[str]] = None,
    excluded_menu_item_ids: Optional[list[str]] = None,
    candidate_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
):
    diets = [_normalise(value) for value in dietary_preferences or [] if value]
    cuisines = [value for value in cuisine_preferences or [] if value]
    dishes = [value for value in menu_item_preferences or [] if value]
    stall_names = [_normalise(value) for value in stall_name_preferences or [] if value]
    excluded_ids = set(excluded_menu_item_ids or [])
    candidates = _filtered_menu(
        candidate_centres=candidate_centres,
        candidate_centre_ids=candidate_centre_ids,
    )

    # Only short-circuit for verifiable dietary restrictions (vegetarian, vegan,
    # halal, gluten-free, etc.) that require structured ingredient/allergen data
    # to confirm. Inferred heuristics such as "healthy", "unhealthy", and
    # "halal" (when derived from Malay/Muslim stall signals) are matched
    # against the catalogue's inferred dietary_tags instead.
    verifiable_diets = [d for d in diets if d in VERIFIABLE_DIETS]
    if verifiable_diets:
        return []

    filtered = []
    for item in candidates:
        if item.get("menu_item_id") in excluded_ids:
            continue
        categories = item["categories"]
        if cuisines and not any(
            _matches_category(preference, categories)
            for preference in cuisines
        ):
            continue
        if stall_names:
            stall_name = _normalise(item["stall_name"])
            if not any(term in stall_name for term in stall_names):
                continue
        if dishes:
            searchable = _normalise(item["menu_item_name"])
            if not all(_normalise(term) in searchable for term in dishes):
                continue
        item_tags = set(item.get("dietary_tags", []))
        matched_diets = [d for d in diets if d in VERIFIABLE_DIETS and d in item_tags]
        if diets and not matched_diets:
            # A verifiable diet was requested but this item does not carry it.
            continue
        filtered.append({
            **item,
            "matched_dietary_tags": sorted(matched_diets),
            "matched_cuisine": [
                preference for preference in cuisines
                if _matches_category(preference, categories)
            ],
            "dietary_suitable": True if matched_diets else None,
            "preference_score": 1.0 if cuisines or dishes else 0.0,
            "reason": "Menu item matched the source category or requested dish name.",
        })
    return filtered


def find_stalls_within_budget(
    budget: Optional[float],
    hawker_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
    excluded_menu_item_ids: Optional[list[str]] = None,
):
    candidates = _filtered_menu(
        candidate_centres=hawker_centres,
        candidate_centre_ids=candidate_centre_ids,
    )
    if budget is None:
        return candidates
    limit = float(budget)
    excluded_ids = set(excluded_menu_item_ids or [])
    matched = []
    for item in candidates:
        if item.get("menu_item_id") in excluded_ids:
            continue
        minimum = item.get("price_sgd_min")
        if minimum is None or minimum > limit:
            continue
        matched.append({
            **item,
            "budget_limit_sgd": limit,
            "budget_match": "priced_within_budget" if item.get("price_sgd_max") is not None and item["price_sgd_max"] <= limit else "starting_price_within_budget",
        })
    return matched


def search_menu_items(
    query: Optional[str] = None,
    hawker_centre_id: Optional[str] = None,
    budget: Optional[float] = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Local menu lookup for the API and chat workers; no LLM catalogue dump."""
    candidates = _filtered_menu(
        candidate_centre_ids=[hawker_centre_id] if hawker_centre_id else None,
    )
    wanted = _normalise(query or "")
    if wanted:
        candidates = [
            item for item in candidates
            if wanted in _normalise(item["menu_item_name"])
            or wanted in _normalise(item["stall_name"])
            or wanted in _normalise(item["hawker_centre"])
            or any(wanted in _normalise(category) for category in item["categories"])
        ]
        add_on_terms = ("add on", "addon", "extra ", "additional ", "side ")
        def relevance(item: dict[str, Any]) -> tuple[int, int, str]:
            name = _normalise(item["menu_item_name"])
            stall = _normalise(item["stall_name"])
            is_add_on = any(term in name for term in add_on_terms)
            if name == wanted:
                rank = 0
            elif wanted in name and not is_add_on:
                rank = 1
            elif wanted in name:
                rank = 2
            elif wanted in stall:
                rank = 3
            elif any(wanted in _normalise(category) for category in item["categories"]):
                rank = 4
            else:
                rank = 5
            return rank, item.get("price_sgd_min") or float("inf"), name

        candidates.sort(key=relevance)
    if budget is not None:
        candidates = [
            item for item in candidates
            if item.get("price_sgd_min") is not None
            and item["price_sgd_min"] <= float(budget)
        ]
    if budget is not None:
        candidates.sort(key=lambda item: (
            relevance(item)[0] if wanted else 0,
            item.get("price_sgd_min") is None,
            item.get("price_sgd_min") if item.get("price_sgd_min") is not None else float("inf"),
            item["menu_item_name"].casefold(),
        ))
    return candidates[:max(1, min(int(limit), 100))]


def list_menu_stalls(
    candidate_centre_ids: Optional[list[str]] = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in _filtered_menu(candidate_centre_ids=candidate_centre_ids):
        stall_id = item["stall_id"]
        stall = grouped.setdefault(stall_id, {
            "stall_id": stall_id,
            "hawker_centre_id": item["hawker_centre_id"],
            "hawker_centre": item["hawker_centre"],
            "stall_name": item["stall_name"],
            "unit_number": item["unit_number"],
            "source_url": item["source_url"],
            "data_origin": item["data_origin"],
            "is_mock": item["is_mock"],
            "mock_notice": item["mock_notice"],
            "menu_items": [],
        })
        stall["menu_items"].append({
            "menu_item_id": item["menu_item_id"],
            "name": item["menu_item_name"],
            "price_label": item["price_label"],
            "price_sgd": item["price_sgd"],
            "price_sgd_min": item["price_sgd_min"],
            "price_sgd_max": item["price_sgd_max"],
            "price_raw": item["price_raw"],
            "price_model": item["price_model"],
            "data_origin": item["data_origin"],
            "is_mock": item["is_mock"],
            "mock_notice": item["mock_notice"],
        })
    stalls = list(grouped.values())
    start = max(0, int(offset))
    return stalls[start:start + max(1, min(int(limit), 500))]
