import json
import os
from typing import Optional

from services.hawker_service import INFERRED_DIETS, find_dietary_stalls


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
    "noodle",
    "rice",
)


def _extract_keywords(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    result = []
    text_lower = text.lower()
    for word in vocabulary:
        if word in text_lower:
            normalized_word = word.replace(" ", "-")
            if normalized_word not in result:
                result.append(normalized_word)
    return result


def _candidate_key(item: dict) -> str:
    if item.get("menu_item_id"):
        return f"id:{item['menu_item_id']}"
    stall_name = str(item.get("stall_name", "")).casefold()
    menu_name = str(item.get("menu_item_name", "")).casefold()
    return f"name:{stall_name}|{menu_name}"


def _fallback_assess(candidates: list[dict], dietary_preferences: list[str]) -> list[dict]:
    output = []
    for item in candidates:
        searchable_text = " ".join(
            [
                str(item.get("stall_name", "")),
                str(item.get("menu_item_name", "")),
                " ".join(item.get("categories", [])),
            ]
        ).casefold()

        matched = []
        for pref in dietary_preferences:
            normalized_pref = pref.replace("-", " ").casefold()
            if normalized_pref in searchable_text:
                matched.append(pref)

        suitable = len(matched) == len(dietary_preferences)
        output.append(
            {
                **item,
                "dietary_suitable": suitable,
                "dietary_confidence": 0.4 if suitable else 0.1,
                "matched_dietary_tags": matched,
                "reason": (
                    "Fallback keyword match from menu metadata."
                    if suitable
                    else "Fallback could not verify dietary suitability."
                ),
            }
        )
    return output


def _llm_assess(
    task: str,
    user_request: str,
    dietary_preferences: list[str],
    candidates: list[dict],
) -> list[dict]:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    compact_candidates = []
    for item in candidates[:100]:
        compact_candidates.append(
            {
                "menu_item_id": item.get("menu_item_id"),
                "stall_name": item.get("stall_name"),
                "menu_item_name": item.get("menu_item_name"),
                "categories": item.get("categories", []),
            }
        )

    prompt = (
        "You are checking dietary suitability.\n"
        f"Task: {task}\n"
        f"User request: {user_request}\n"
        f"Dietary preferences: {dietary_preferences}\n"
        f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}\n\n"
        "Return JSON ONLY with this shape:\n"
        "{\"assessments\":[{\"menu_item_id\":\"...\",\"stall_name\":\"...\","
        "\"menu_item_name\":\"...\",\"dietary_suitable\":true,"
        "\"confidence\":0.8,\"reason\":\"...\"}]}\n"
        "Use only given metadata. If uncertain, set dietary_suitable to false."
    )

    response_text = model.invoke(prompt).content
    parsed = json.loads(response_text)
    assessments = parsed.get("assessments", [])

    assessed_by_key = {}
    for item in assessments:
        menu_item_id = item.get("menu_item_id")
        if menu_item_id:
            key = f"id:{menu_item_id}"
        else:
            key = (
                f"name:{str(item.get('stall_name', '')).casefold()}"
                f"|{str(item.get('menu_item_name', '')).casefold()}"
            )
        assessed_by_key[key] = {
            "dietary_suitable": bool(item.get("dietary_suitable", False)),
            "dietary_confidence": float(item.get("confidence", 0.0)),
            "reason": str(item.get("reason", "No reason returned by LLM.")),
        }

    merged = []
    for item in candidates:
        key = _candidate_key(item)
        llm_result = assessed_by_key.get(key)
        if llm_result:
            merged.append({**item, **llm_result})
        else:
            merged.append(
                {
                    **item,
                    "dietary_suitable": False,
                    "dietary_confidence": 0.0,
                    "reason": "No assessment returned for this item.",
                }
            )
    return merged


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
    for item in inferred_dietary:
        if item not in final_dietary:
            final_dietary.append(item)

    final_cuisine = list(cuisine_preferences or [])
    for item in inferred_cuisine:
        if item not in final_cuisine:
            final_cuisine.append(item)

    # Pull source-backed candidates first. Dietary suitability is decided below.
    stalls = find_dietary_stalls(
        dietary_preferences=[],
        cuisine_preferences=final_cuisine,
        menu_item_preferences=menu_item_preferences,
        stall_name_preferences=stall_name_preferences,
        excluded_menu_item_ids=excluded_menu_item_ids,
        candidate_centres=candidate_centres,
        candidate_centre_ids=candidate_centre_ids,
    )

    limitations = []
    confidence = 0.3
    scored_stalls = list(stalls)
    inferred_diets = [diet for diet in final_dietary if diet in INFERRED_DIETS]

    if final_dietary:
        try:
            if os.getenv("OPENAI_API_KEY"):
                scored_stalls = _llm_assess(
                    task=task,
                    user_request=user_request or task,
                    dietary_preferences=final_dietary,
                    candidates=stalls,
                )
            else:
                scored_stalls = _fallback_assess(stalls, final_dietary)
                limitations.append(
                    "OPENAI_API_KEY is not set, so dietary suitability used fallback logic."
                )
        except Exception as error:
            scored_stalls = _fallback_assess(stalls, final_dietary)
            limitations.append(
                f"LLM dietary assessment failed ({error}); fallback logic was used."
            )

        limitations.append(
            "The menu source has no ingredient, allergen, or certified dietary fields; "
            "dietary suitability cannot be verified."
        )
        suitable = [item for item in scored_stalls if item.get("dietary_suitable") is True]
        confidence = 0.9 if suitable else 0.6
    elif final_cuisine:
        confidence = 0.9 if scored_stalls else 0.5
    else:
        confidence = 0.6 if scored_stalls else 0.3
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
        "candidates": scored_stalls,
        "limitations": limitations,
    }
