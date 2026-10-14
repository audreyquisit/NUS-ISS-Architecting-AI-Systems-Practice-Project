from typing import Optional

from services.hawker_service import find_dietary_stalls, INFERRED_DIETS, VERIFIABLE_DIETS


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
    "healthy",
    "unhealthy",
)

# Terms that imply a halal preference in the Singapore hawker context.
HALAL_SYNONYMS = ("malay", "muslim")

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
    source_text = " ".join([task or "", user_request or ""]).strip()

    inferred_dietary = _extract_keywords(source_text, DIETARY_KEYWORDS)
    inferred_cuisine = _extract_keywords(source_text, CUISINE_KEYWORDS)

    # Map halal synonyms (Malay / Muslim) onto the canonical halal tag so the
    # catalogue's inferred halal tag can match the user's intent.
    for synonym in HALAL_SYNONYMS:
        if synonym in source_text and "halal" not in inferred_dietary:
            inferred_dietary.append("halal")

    final_dietary = list(dietary_preferences or [])
    for pref in inferred_dietary:
        if pref not in final_dietary:
            final_dietary.append(pref)

    final_cuisine = list(cuisine_preferences or [])
    for pref in inferred_cuisine:
        if pref not in final_cuisine:
            final_cuisine.append(pref)

    # Separate verifiable restrictions (which need ingredient/allergen data
    # the source lacks) from inferred heuristics (halal, healthy, unhealthy)
    # that are matched against the stall's inferred dietary_tags.
    verifiable_diets = [
        d for d in final_dietary
        if d in VERIFIABLE_DIETS and d not in INFERRED_DIETS
    ]
    inferred_diets = [d for d in final_dietary if d in INFERRED_DIETS]

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
    confidence = 0.3
    if stalls:
        if final_dietary:
            suitable = [item for item in stalls if item.get(
                "dietary_suitable") is True]
            confidence = 0.9 if suitable else 0.6
        elif final_cuisine:
            confidence = 0.9
        else:
            confidence = 0.6
    if verifiable_diets:
        limitations.append(
            "The menu source has no ingredient, allergen, or certified dietary fields; "
            "dietary suitability cannot be verified."
        )
    if inferred_diets:
        limitations.append(
            "Dietary suitability is inferred from stall name and category heuristics "
            "(Malay/Muslim -> halal; Yong Tau Foo/vegetables -> healthy; "
            "fried items/desserts -> unhealthy), not from structured certification data."
        )
    if candidate_centres and not stalls and not final_dietary:
        limitations.append(
            "No menu items matched the selected centres and dish or cuisine preferences."
        )
    return {
        "agent": "dietary",
        "confidence": confidence,
        "task": task,
        "preferences": {
            "dietary": final_dietary,
            "cuisine": final_cuisine,
        },
        "candidates": stalls,
        "limitations": limitations,
    }
