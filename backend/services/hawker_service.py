from typing import Optional


def _stall_catalog():
    return [
        {
            "stall_name": "Example Vegetarian Stall",
            "hawker_centre": "Example Hawker Centre",
            "distance_minutes": 5,
            "price": 6.50,
            "cuisine": "vegetarian",
            "dietary_tags": ["vegetarian", "vegan", "no-pork", "no-beef"],
        },
        {
            "stall_name": "Example Noodle Stall",
            "hawker_centre": "Example Hawker Centre",
            "distance_minutes": 8,
            "price": 7.00,
            "cuisine": "chinese",
            "dietary_tags": ["contains-egg"],
        },
        {
            "stall_name": "Halal Rice Corner",
            "hawker_centre": "Example Hawker Centre",
            "distance_minutes": 6,
            "price": 7.50,
            "cuisine": "malay",
            "dietary_tags": ["halal", "no-pork"],
        },
    ]


def find_nearby_stalls(location: str):
    return [
        {
            "stall_name": stall["stall_name"],
            "hawker_centre": stall["hawker_centre"],
            "distance_minutes": stall["distance_minutes"],
            "price": stall["price"],
            "cuisine": stall["cuisine"],
        }
        for stall in _stall_catalog()
    ]


def find_dietary_stalls(
    dietary_preferences: Optional[list[str]] = None,
    cuisine_preferences: Optional[list[str]] = None,
):
    normalized_dietary_preferences = []
    for value in dietary_preferences or []:
        if not value:
            continue
        normalized_value = value.strip().lower()
        if normalized_value:
            normalized_dietary_preferences.append(normalized_value)

    normalized_cuisine_preferences = []
    for value in cuisine_preferences or []:
        if not value:
            continue
        normalized_value = value.strip().lower()
        if normalized_value:
            normalized_cuisine_preferences.append(normalized_value)

    candidates = []

    for stall in _stall_catalog():
        tags = []
        for tag in stall.get("dietary_tags", []):
            if not tag:
                continue
            normalized_tag = tag.strip().lower()
            if normalized_tag:
                tags.append(normalized_tag)

        cuisine = str(stall.get("cuisine", "")).strip().lower()

        matched_dietary = []
        for preference in normalized_dietary_preferences:
            if preference in tags:
                matched_dietary.append(preference)

        matched_cuisine = False
        if normalized_cuisine_preferences:
            if cuisine in normalized_cuisine_preferences:
                matched_cuisine = True

        dietary_ok = False
        if not normalized_dietary_preferences:
            dietary_ok = True
        elif len(matched_dietary) == len(normalized_dietary_preferences):
            dietary_ok = True

        preference_hits = len(matched_dietary) + (1 if matched_cuisine else 0)
        preference_base = len(normalized_dietary_preferences)
        if normalized_cuisine_preferences:
            preference_base += 1
        if preference_base < 1:
            preference_base = 1

        preference_score = round(preference_hits / preference_base, 2)

        if dietary_ok and matched_cuisine:
            reason = "Matches dietary constraints and preferred cuisine."
        elif dietary_ok:
            reason = "Matches dietary constraints."
        else:
            reason = "Does not satisfy all dietary constraints."

        candidates.append(
            {
                "stall_name": stall["stall_name"],
                "hawker_centre": stall["hawker_centre"],
                "cuisine": stall["cuisine"],
                "price": stall["price"],
                "dietary_tags": tags,
                "matched_dietary_tags": matched_dietary,
                "dietary_suitable": dietary_ok,
                "preference_score": preference_score,
                "reason": reason,
            }
        )

    candidates.sort(
        key=lambda item: (
            not item["dietary_suitable"],
            -item["preference_score"],
            item["price"],
        )
    )
    return candidates


def find_stalls_within_budget(
    budget: float
):
    return [
        {
            "stall_name": stall["stall_name"],
            "price": stall["price"],
        }
        for stall in _stall_catalog()
        if stall["price"] <= budget
    ]
