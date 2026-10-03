"""Fill NEA centres without stall coverage with clearly labelled demo listings."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"
REAL_STALLS_PATH = DATA_DIR / "hawker_stall.json"
MOCK_STALLS_PATH = DATA_DIR / "hawker_stall_mock.json"

DISHES = (
    {"key": "chicken_rice", "name": "Chicken rice", "price_range": (3.50, 6.00)},
    {"key": "wanton_mee", "name": "Wanton mee", "price_range": (3.50, 4.50)},
    {"key": "hokkien_mee", "name": "Hokkien mee", "price_range": (4.00, 7.00)},
)


def _price_for(centre_id: str, dish_key: str, low: float, high: float) -> float:
    seed_bytes = hashlib.sha256(f"{centre_id}:{dish_key}".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    steps = int(round((high - low) / 0.5))
    return round(low + rng.randint(0, steps) * 0.5, 2)


def _mock_stall(centre: dict[str, Any], dish: dict[str, Any]) -> dict[str, Any]:
    price = _price_for(
        centre["id"], dish["key"], *dish["price_range"]
    )
    price_text = f"${price:.2f}"
    stall_id = f"demo_{centre['id']}_{dish['key']}"
    menu_item_id = f"{stall_id}_item_000"
    name = f"Demo {dish['name']} Stall"
    return {
        "url": None,
        "name": name,
        "category": "Chinese",
        "hawker_centre": centre["name"],
        "location": "Demo listing; no real unit number",
        "address": centre.get("address"),
        "menu_items": [{
            "name": dish["name"],
            "price": price_text,
            "menu_item_id": menu_item_id,
            "price_raw": price_text,
            "price_sgd": price,
            "price_sgd_min": price,
            "price_sgd_max": price,
            "price_sgd_options": [price],
            "price_option_labels": [],
            "price_status": "fixed",
            "price_basis": "illustrative_only",
            "price_model": "fixed",
        }],
        "menu_text": f"{dish['name']} {price_text} (illustrative demo price)",
        "menu_images": [],
        "stall_id": stall_id,
        "hawker_centre_id": centre["id"],
        "data_origin": "synthetic_mock",
        "is_mock": True,
        "mock_notice": (
            "Fictional demo stall, menu, and price; not verified at this centre."
        ),
    }


def generate() -> dict[str, int]:
    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    real_catalogue = json.loads(REAL_STALLS_PATH.read_text(encoding="utf-8"))
    current = json.loads(MOCK_STALLS_PATH.read_text(encoding="utf-8")) if MOCK_STALLS_PATH.exists() else {
        "source": "Synthetic SmartHawker demo data; not an official or observed stall directory.",
        "is_mock": True,
        "stalls": [],
    }

    real_ids = {
        item.get("hawker_centre_id")
        for item in real_catalogue.get("stalls", [])
        if item.get("hawker_centre_id")
    }
    existing_by_id = {
        item.get("stall_id"): item for item in current.get("stalls", [])
    }
    generated_centres = set()
    for centre in centres:
        if centre["id"] in real_ids:
            continue
        generated_centres.add(centre["id"])
        for dish in DISHES:
            stall = _mock_stall(centre, dish)
            existing_by_id[stall["stall_id"]] = stall

    current["stalls"] = list(existing_by_id.values())
    current["normalization"] = {
        "coverage": "Three illustrative demo dishes are generated for NEA centres without real catalogue coverage.",
        "price_currency": "SGD",
        "price_note": "All demo prices are synthetic examples and are not source-derived or verified.",
    }
    MOCK_STALLS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "nea_centres": len(centres),
        "centres_missing_real_or_mock_stalls_before_generation": len(generated_centres),
        "mock_stalls": len(current["stalls"]),
        "mock_menu_items": sum(len(stall.get("menu_items", [])) for stall in current["stalls"]),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
