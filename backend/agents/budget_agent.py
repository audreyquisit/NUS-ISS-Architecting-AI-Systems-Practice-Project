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
    if budget is not None and not stalls:
        limitations.append(
            "No menu item with a parseable listed starting price matched the budget "
            "and selected centres."
        )
    if budget is not None:
        limitations.append(
            "Budget matching is per listed menu item; it does not estimate the total cost of a full meal."
        )

    return {
        "agent": "budget",
        "task": task,
        "budget_limit": budget,
        "candidates": stalls,
        "limitations": limitations,
    }
