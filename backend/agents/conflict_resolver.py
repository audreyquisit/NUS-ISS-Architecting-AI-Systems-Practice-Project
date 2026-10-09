"""Resolve conflicts between specialised agents by re-ranking joined candidates."""

from functools import lru_cache
from typing import Any

from services.hawker_service import _menu_catalogue

_PRIORITY_KEYWORDS = {
    "budget": ("budget", "price", "cheap", "affordable"),
    "location": ("location", "near", "nearby", "close", "convenient", "walking", "walk"),
    "queue": ("queue", "wait", "waiting", "crowd", "crowded"),
    "dietary": ("dietary", "halal", "vegetarian", "vegan", "healthy"),
    "weather": ("weather", "rain", "rainy", "outdoor", "shelter"),
}

_FAIRNESS_WEIGHT = 0.05


def _weights(priorities: list[str]) -> dict[str, float]:
    weights = {
        "budget": 0.0,
        "location": 0.0,
        "queue": 0.0,
        "dietary": 0.0,
        "weather": 0.0,
    }
    if not priorities:
        return weights
    for priority in priorities:
        for key, keywords in _PRIORITY_KEYWORDS.items():
            if any(keyword in priority.lower() for keyword in keywords):
                weights[key] = max(weights[key], 0.3)
    total = sum(weights.values())
    if total == 0:
        return weights
    for key in weights:
        weights[key] /= total
    return weights


@lru_cache(maxsize=1)
def _menu_count_by_stall() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _menu_catalogue():
        stall_id = item.get("stall_id")
        if not stall_id:
            continue
        counts[stall_id] = counts.get(stall_id, 0) + 1
    return counts


def _score(candidate: dict[str, Any], weights: dict[str, float]) -> float:
    price = candidate.get("price_sgd_min") or candidate.get("price_sgd_max") or 999.0
    travel = (candidate.get("travel") or {}).get("total_duration_min") or 999.0
    queue = candidate.get("queue_minutes") or 999.0
    dietary = 1.0 if candidate.get("dietary_suitable") is True else 0.5
    weather = candidate.get("weather_score") or 0.5

    score = (
        weights["budget"] * max(0.0, 1.0 - price / 30.0)
        + weights["location"] * max(0.0, 1.0 - travel / 120.0)
        + weights["queue"] * max(0.0, 1.0 - queue / 60.0)
        + weights["dietary"] * dietary
        + weights["weather"] * weather
    )

    stall_id = candidate.get("stall_id")
    if stall_id and weights.get("queue", 0.0) > 0:
        menu_counts = _menu_count_by_stall()
        menu_count = menu_counts.get(stall_id, 0)
        max_count = max(menu_counts.values()) if menu_counts else 1
        popularity = menu_count / max_count if max_count else 0.0
        score += _FAIRNESS_WEIGHT * (1.0 - popularity)

    return score


def resolve_conflicts(
    verified_results: list[dict[str, Any]],
    parsed_request: dict[str, Any],
) -> list[dict[str, Any]]:
    priorities = parsed_request.get("priorities", [])
    if not priorities:
        return verified_results

    joined = next(
        (item for item in verified_results if item.get("agent") == "candidate_join"),
        None,
    )
    if not joined or not joined.get("candidates"):
        return verified_results

    candidates = list(joined["candidates"])
    weights = _weights(priorities)

    if all(v == 0.0 for v in weights.values()):
        return verified_results

    scored = sorted(
        ((_score(candidate, weights), candidate) for candidate in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    new_candidates = [candidate for _, candidate in scored]

    updated = list(verified_results)
    for index, item in enumerate(updated):
        if item.get("agent") == "candidate_join":
            updated[index] = {**item, "candidates": new_candidates}
            break

    note = (
        "Re-ranked by priorities: "
        + ", ".join(priorities)
        + ". Weights applied: "
        + ", ".join(f"{k}={v:.2f}" for k, v in weights.items() if v > 0)
        + "."
    )
    updated.append({
        "agent": "conflict_resolution",
        "verified": True,
        "candidates": [],
        "limitations": [],
        "verification": note,
    })
    return updated
