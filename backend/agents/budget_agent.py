from typing import Optional

from services.hawker_service import find_stalls_within_budget


def run(
    task: str,
    budget: Optional[float] = None,
    excluded_menu_item_ids: Optional[list[str]] = None,
    candidate_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
):

    print("Budget Agent")

    stalls = find_stalls_within_budget(
        budget=budget,
        excluded_menu_item_ids=excluded_menu_item_ids,
        hawker_centres=candidate_centres,
        candidate_centre_ids=candidate_centre_ids,
    )

    limitations = []
    confidence = 0.3
    if budget is not None:
        if len(stalls) >= 3:
            under = [s for s in stalls if (s.get("price_sgd_min") or 999) <= budget * 0.8]
            confidence = 0.9 if under else 0.6
        elif stalls:
            confidence = 0.6
        if not stalls:
            confidence = 0.3
        limitations.append(
            "Budget matching is per listed menu item; it does not estimate the total cost of a full meal."
        )
    if budget is not None and not stalls:
        limitations.append(
            "No menu item with a parseable listed starting price matched the budget "
            "and selected centres."
        )
    return {
        "agent": "budget",
        "confidence": confidence,
        "task": task,
        "budget_limit": budget,
        "candidates": stalls,
        "limitations": limitations,
    }
