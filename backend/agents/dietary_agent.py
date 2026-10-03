from typing import Optional

from services.hawker_service import find_dietary_stalls


DIETARY_KEYWORDS = (
    "vegetarian",
    "vegan",
    "halal",
    "no-pork",
    "no beef",
    "no-beef",
    "gluten-free",
    "dairy-free",
    "nut-free",
    "seafood-free",
)

CUISINE_KEYWORDS = (
    "chinese",
    "malay",
    "indian",
    "western",
)


def _extract_keywords(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    extracted = []
    lowered = text.lower()
    for keyword in vocabulary:
        if keyword in lowered and keyword not in extracted:
            normalized = keyword.replace(" ", "-")
            extracted.append(normalized)
    return extracted


def run(
    task: str,
    user_request: Optional[str] = None,
    dietary_preferences: Optional[list[str]] = None,
    cuisine_preferences: Optional[list[str]] = None,
    menu_item_preferences: Optional[list[str]] = None,
    stall_name_preferences: Optional[list[str]] = None,
    excluded_menu_item_ids: Optional[list[str]] = None,
    candidate_centres: Optional[list[str]] = None,
    candidate_centre_ids: Optional[list[str]] = None,
):
    print("Dietary Agent")

    source_text = " ".join([task or "", user_request or ""]).strip()

    inferred_dietary = _extract_keywords(source_text, DIETARY_KEYWORDS)
    inferred_cuisine = _extract_keywords(source_text, CUISINE_KEYWORDS)

    final_dietary = list(dietary_preferences or [])
    for pref in inferred_dietary:
        if pref not in final_dietary:
            final_dietary.append(pref)

    final_cuisine = list(cuisine_preferences or [])
    for pref in inferred_cuisine:
        if pref not in final_cuisine:
            final_cuisine.append(pref)

    stalls = find_dietary_stalls(
        dietary_preferences=final_dietary,
        cuisine_preferences=final_cuisine,
        menu_item_preferences=menu_item_preferences,
        stall_name_preferences=stall_name_preferences,
        excluded_menu_item_ids=excluded_menu_item_ids,
        candidate_centres=candidate_centres,
        candidate_centre_ids=candidate_centre_ids,
    )

    limitations = []
    if final_dietary:
        limitations.append(
            "The menu source has no ingredient, allergen, or certified dietary fields; "
            "dietary suitability cannot be verified."
        )
    if candidate_centres and not stalls and not final_dietary:
        limitations.append(
            "No menu items matched the selected centres and dish or cuisine preferences."
        )
    return {
        "agent": "dietary",
        "task": task,
        "preferences": {
            "dietary": final_dietary,
            "cuisine": final_cuisine,
        },
        "candidates": stalls,
        "limitations": limitations,
    }
