"""Add stable IDs, NEA centre links, and parsed SGD price fields to the menu data."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.source_matching import CentreMatcher, extract_postcode  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"

# The day this catalogue was last refreshed in the repository. The source
# itself does not state a publication date, so the repository is what the
# date is taken from.
CATALOGUE_AS_OF = "2026-10-13"

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


def _stall_id(stall: dict[str, Any]) -> str:
    tail = str(stall.get("url", "")).rstrip("/").rsplit("/", 1)[-1]
    return f"wakwak_{tail}" if tail else ""


def _street_of(address: str) -> str | None:
    """Return the street part of a stall address, used to break postcode ties."""
    cleaned = re.sub(r",\s*Singapore\s*\d{6}.*$", "", address or "", flags=re.I)
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) < 2:
        return None
    return parts[1]


def normalize_catalogue() -> dict[str, int]:
    catalogue = json.loads(STALLS_PATH.read_text(encoding="utf-8"))
    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    matcher = CentreMatcher(centres)

    mapped_stalls = 0
    match_methods: dict[str, int] = defaultdict(int)
    unmatched: dict[str, int] = defaultdict(int)
    price_statuses: dict[str, int] = defaultdict(int)
    for stall in catalogue.get("stalls", []):
        stall["stall_id"] = _stall_id(stall)
        address = str(stall.get("address") or "")
        centre, method = matcher.match(
            postcode=extract_postcode(address),
            centre_name=stall.get("hawker_centre"),
            street=_street_of(address),
            address=address,
        )
        stall["hawker_centre_id"] = centre["id"] if centre else None
        stall["centre_match_method"] = method
        stall["data_as_of"] = CATALOGUE_AS_OF
        if centre:
            mapped_stalls += 1
            match_methods[method] += 1
        else:
            unmatched[str(stall.get("hawker_centre") or "unknown")] += 1

        for index, item in enumerate(stall.get("menu_items", [])):
            item["menu_item_id"] = f"{stall['stall_id']}_item_{index:03d}"
            normalized = normalize_price(item.get("price"), item.get("name"))
            item.update(normalized)
            price_statuses[normalized["price_status"]] += 1

    catalogue["as_of"] = CATALOGUE_AS_OF
    catalogue["normalization"] = {
        "price_currency": "SGD",
        "hawker_centre_id_source": (
            "Exact postal-code match to hawker_centres.json, then a unique centre "
            "name match, then a postal-code match narrowed by street, then a "
            "street match that only one centre has. Unmatched values remain "
            "null and are counted in the coverage report."
        ),
        "price_fields": "Original price is retained in price_raw and price. Numeric fields are populated only from explicit amounts.",
    }
    catalogue["unmatched_centres"] = {
        "count": sum(unmatched.values()),
        "by_centre_name": dict(sorted(unmatched.items())),
    }
    STALLS_PATH.write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "stalls": len(catalogue.get("stalls", [])),
        "mapped_stalls": mapped_stalls,
        "unmatched_stalls": sum(unmatched.values()),
        "menu_items": sum(len(s.get("menu_items", [])) for s in catalogue.get("stalls", [])),
        **{f"match_{key}": value for key, value in sorted(match_methods.items())},
        **{f"price_{key}": value for key, value in sorted(price_statuses.items())},
    }


if __name__ == "__main__":
    print(json.dumps(normalize_catalogue(), indent=2))
