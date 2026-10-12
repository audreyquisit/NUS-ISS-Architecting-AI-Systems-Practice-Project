"""Infer dietary tags from stall name and category fields.

The Wak-Wak catalogue has no structured ingredient, allergen, or certification
data, so the tags produced here are *inferred heuristics*, not verified facts.
They are additive and idempotent: re-running preserves tags added by other
sources (such as the HPB Healthier Dining match in ``enrich_dietary_tags.py``).

Inference rules (from project guidance):

* Any stall whose category or name mentions Malay / Muslim is tagged
  ``halal`` (a halal dietary tag).
* Yong Tau Foo and vegetable-centric dishes are tagged ``healthy``.
* Fried items and desserts / snacks are tagged ``unhealthy``.

Usage (from backend/):
    python scripts/infer_dietary_tags.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure stdout can handle non-ASCII characters from stall names.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
MOCK_STALLS_PATH = DATA_DIR / "hawker_stall_mock.json"

# Tags managed by this script. Tags from other sources are preserved.
MANAGED_TAGS = frozenset({"halal", "healthy", "unhealthy"})

# Category tokens that imply halal in the Singapore hawker context.
HALAL_CATEGORY_TOKENS = frozenset({"malay", "muslim"})
# Name tokens that imply halal.
HALAL_NAME_TOKENS = frozenset({"malay", "muslim", "halal"})

# Categories / dishes treated as healthy by project guidance.
HEALTHY_CATEGORY_TOKENS = frozenset({
    "yong tau foo", "vegetable", "vegetarian", "salad", "tofu",
})
HEALTHY_NAME_TERMS = ("yong tau foo", "vegetable", "vegetarian", "salad")

# Categories / dishes treated as unhealthy by project guidance.
UNHEALTHY_CATEGORY_TOKENS = frozenset({"dessert", "snack", "dessert / snack"})
UNHEALTHY_KEYWORDS = (
    "fried", "deep fried", "deep-fried", "batter", "fritter",
)


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _category_tokens(category: str) -> set[str]:
    """Split the comma/slash-separated category field into normalised tokens."""
    return set(
        token
        for token in re.split(r"[,/]", _normalise(category))
        if token
    )


def _infer_tags(stall: dict[str, Any]) -> set[str]:
    """Return the set of dietary tags inferred for a single stall."""
    tags: set[str] = set()
    category = str(stall.get("category") or "")
    name = str(stall.get("name") or "")
    cat_tokens = _category_tokens(category)
    name_tokens = set(_normalise(name).split())
    text = f"{_normalise(category)} {_normalise(name)}"

    # Halal: any Malay / Muslim signal in the category or stall name.
    if (HALAL_CATEGORY_TOKENS & cat_tokens) or (HALAL_NAME_TOKENS & name_tokens):
        tags.add("halal")

    # Healthy: Yong Tau Foo / vegetable-centric dishes.
    if "yong tau foo" in cat_tokens or any(term in text for term in HEALTHY_NAME_TERMS):
        tags.add("healthy")

    # Unhealthy: fried items and desserts / snacks.
    if UNHEALTHY_CATEGORY_TOKENS & cat_tokens:
        tags.add("unhealthy")
    if any(keyword in text for keyword in UNHEALTHY_KEYWORDS):
        tags.add("unhealthy")

    return tags


def apply_inference(stalls: list[dict[str, Any]]) -> int:
    """Add inferred tags to each stall in place. Returns the number tagged."""
    tagged = 0
    for stall in stalls:
        inferred = _infer_tags(stall)
        if not inferred:
            continue
        existing = set(stall.get("dietary_tags", []))
        new_tags = inferred - existing
        if new_tags:
            stall["dietary_tags"] = sorted(existing | inferred)
            tagged += 1
            print(f"  Inferred {sorted(new_tags)}: {stall.get('name', '')}")
    return tagged


def _enrich_file(path: Path) -> int:
    if not path.exists():
        print(f"Skipping missing file: {path.name}")
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    stalls = data.get("stalls", [])
    count = apply_inference(stalls)
    data["stalls"] = stalls
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Tagged {count} stalls in {path.name}")
    return count


def main() -> None:
    total = 0
    total += _enrich_file(STALLS_PATH)
    total += _enrich_file(MOCK_STALLS_PATH)
    print(f"\nTotal stalls with inferred dietary tags: {total}")


if __name__ == "__main__":
    main()