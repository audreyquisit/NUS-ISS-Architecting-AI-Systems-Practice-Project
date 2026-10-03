"""Add stable IDs, NEA centre links, and parsed SGD price fields to the menu data."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"

PRICE_TOKEN = re.compile(r"\$\s*(\d+(?:[.,]\d{1,2})?)|(?<![\w.])(\d+(?:[.,]\d{1,2})?)\s*SGD\b", re.I)
PLAIN_PRICE = re.compile(r"^\s*(\d+(?:[.,]\d{1,2})?)\s*$")


def _amount(value: str) -> float:
    # The scrape uses both decimal points and decimal commas (for example $1,50).
    return round(float(value.replace(",", ".")), 2)


def normalize_price(raw_price: Any, menu_name: Any = None) -> dict[str, Any]:
    raw = str(raw_price or "").strip()
    lowered = raw.casefold()
    name = str(menu_name or "").casefold()
    values = [_amount(a or b) for a, b in PRICE_TOKEN.findall(raw)]
    # Some source rows use shorthand such as "$3/4" to mean $3 or $4.
    # The currency marker applies to every slash-separated option.
    shorthand = re.search(
        r"\$\s*(\d+(?:[.,]\d{1,2})?(?:\s*/\s*\d+(?:[.,]\d{1,2})?)+)",
        raw,
    )
    if shorthand and len(values) == 1:
        values = [
            _amount(value)
            for value in re.findall(r"\d+(?:[.,]\d{1,2})?", shorthand.group(1))
        ]
    if not values:
        plain = PLAIN_PRICE.fullmatch(raw)
        if plain:
            values = [_amount(plain.group(1))]

    values = list(dict.fromkeys(values))
    variable_price = any(term in f"{lowered} {name}" for term in (
        "final price determined", "own choice of ingredients",
        "price depends on ingredients", "price based on ingredients",
    ))
    if variable_price:
        status = "variable"
        values = []
    elif any(term in lowered for term in ("market price", "assorted prices", "prices vary")):
        status = "unavailable"
        values = []
    elif not values:
        status = "unparsed" if raw else "unavailable"
    elif len(values) == 1:
        status = "fixed"
    else:
        status = "options"

    # A three-price slash list without explicit flavour/variant labels is the
    # source's portion convention: smallest, medium, largest.
    portion_tiers = (
        status == "options"
        and len(values) in {2, 3}
        and not re.search(r"\([^)]*(?:/|,|\bor\b)[^)]*\)", str(menu_name or ""), re.I)
        and " for " not in lowered
    )
    portion_labels = (
        ["small", "large"] if portion_tiers and len(values) == 2
        else ["small", "medium", "large"] if portion_tiers
        else []
    )

    basis = "unknown"
    if variable_price:
        basis = "by_selected_ingredients"
    elif re.search(r"(?:\bper\s+portion\b|/\s*portion\b)", lowered):
        basis = "per_portion"
    elif re.search(r"(?:\bper\s+100\s*g\b|/\s*100\s*g\b)", lowered):
        basis = "per_100g"
    elif re.search(r"\bper\s+box\b|\bper\s+box", lowered):
        basis = "per_box"
    elif re.search(r"\bper\s+(?:pc|piece)\b", lowered):
        basis = "per_piece"
    elif re.search(r"\bper\s+(?:bowl|plate|serving|portion)\b", lowered):
        basis = "per_serving"

    return {
        "price_raw": raw or None,
        "price_sgd": values[0] if status == "fixed" else None,
        "price_sgd_min": min(values) if values else None,
        "price_sgd_max": max(values) if values else None,
        "price_sgd_options": values,
        "price_option_labels": portion_labels,
        "price_model": (
            "ingredient_based" if variable_price
            else "portion_tiers" if portion_tiers
            else "fixed" if status == "fixed"
            else "listed_options" if status == "options"
            else "unknown"
        ),
        "price_status": status,
        "price_basis": basis,
    }


def _centre_ids_by_postcode(centres: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_postcode: dict[str, set[str]] = defaultdict(set)
    for centre in centres:
        for postcode in re.findall(r"\b\d{6}\b", centre.get("address", "")):
            by_postcode[postcode].add(centre["id"])
    return by_postcode


def _stall_id(stall: dict[str, Any]) -> str:
    tail = str(stall.get("url", "")).rstrip("/").rsplit("/", 1)[-1]
    return f"wakwak_{tail}" if tail else ""


def normalize_catalogue() -> dict[str, int]:
    catalogue = json.loads(STALLS_PATH.read_text(encoding="utf-8"))
    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    postcodes = _centre_ids_by_postcode(centres)

    mapped_stalls = 0
    price_statuses: dict[str, int] = defaultdict(int)
    for stall in catalogue.get("stalls", []):
        postcode_match = re.search(r"\b(\d{6})\b", stall.get("address", ""))
        centre_ids = postcodes.get(postcode_match.group(1), set()) if postcode_match else set()
        stall["stall_id"] = _stall_id(stall)
        stall["hawker_centre_id"] = next(iter(centre_ids)) if len(centre_ids) == 1 else None
        if stall["hawker_centre_id"]:
            mapped_stalls += 1

        for index, item in enumerate(stall.get("menu_items", [])):
            item["menu_item_id"] = f"{stall['stall_id']}_item_{index:03d}"
            normalized = normalize_price(item.get("price"), item.get("name"))
            item.update(normalized)
            price_statuses[normalized["price_status"]] += 1

    catalogue["normalization"] = {
        "price_currency": "SGD",
        "hawker_centre_id_source": "Exact postal-code match to hawker_centres.json; unmatched values remain null.",
        "price_fields": "Original price is retained in price_raw and price. Numeric fields are populated only from explicit amounts.",
    }
    STALLS_PATH.write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "stalls": len(catalogue.get("stalls", [])),
        "mapped_stalls": mapped_stalls,
        "menu_items": sum(len(s.get("menu_items", [])) for s in catalogue.get("stalls", [])),
        **{f"price_{key}": value for key, value in sorted(price_statuses.items())},
    }


if __name__ == "__main__":
    print(json.dumps(normalize_catalogue(), indent=2))
