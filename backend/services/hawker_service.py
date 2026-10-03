"""Read and locally filter the scraped stall and menu catalogue."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from services.location_service import load_hawker_centres


STALLS_FILE = Path(__file__).resolve().parents[1] / "data" / "hawker_stall.json"
MOCK_STALLS_FILE = Path(__file__).resolve().parents[1] / "data" / "hawker_stall_mock.json"


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


@lru_cache(maxsize=1)
def _menu_catalogue() -> tuple[dict[str, Any], ...]:
    payload = json.loads(STALLS_FILE.read_text(encoding="utf-8"))
    mock_payload = (
        json.loads(MOCK_STALLS_FILE.read_text(encoding="utf-8"))
        if MOCK_STALLS_FILE.exists() else {"stalls": []}
    )
    centres = {item["id"]: item for item in load_hawker_centres()}
    records = []
    for stall in payload.get("stalls", []) + mock_payload.get("stalls", []):
        centre_id = stall.get("hawker_centre_id")
        centre = centres.get(centre_id)
        # The location agent only recommends centres in the NEA catalogue.
        # Keep unmatched source records in the JSON, but don't present them as
        # verified hawker-centre options.
        if not centre:
            continue
        categories = [
            part.strip()
            for part in str(stall.get("category") or "").split(",")
            if part.strip()
        ]
        unit = re.search(r"#\s*([A-Za-z0-9/-]+)", str(stall.get("location") or ""))
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
                # The source has no ingredient, allergen or dietary claim fields.
                "dietary_tags": [],
                "dietary_suitable": None,
                "source_url": stall.get("url"),
                "data_origin": stall.get("data_origin", "wak_wak_source"),
                "is_mock": bool(stall.get("is_mock", False)),
                "mock_notice": stall.get("mock_notice"),
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

    # Never derive allergy/halal/vegetarian claims from a dish or stall name.
    # This catalogue has no structured dietary evidence to verify those claims.
    if diets:
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
        filtered.append({
            **item,
            "matched_dietary_tags": [],
            "matched_cuisine": [
                preference for preference in cuisines
                if _matches_category(preference, categories)
            ],
            "dietary_suitable": None if not diets else False,
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
