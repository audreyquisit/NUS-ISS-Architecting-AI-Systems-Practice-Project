"""Location & Logistics reasoning over verified OneMap route evidence."""

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from models.location_models import (
    CentreLogistics,
    LocationDecision,
    LocationQuery,
    RankedCentre,
    TravelEstimate,
)
from services.location_service import (
    LocationServiceError,
    get_candidate_centres,
    get_route,
    geocode_place,
    is_singapore_location,
    load_hawker_centres,
)


class LocationResolutionError(ValueError):
    """No usable origin or pilot-centre evidence could be established."""


class LocationIntent(BaseModel):
    request_intent: Literal[
        "find_nearby_centres", "explore_centre", "directions_to_centre"
    ] = Field(description="Whether this turn asks for nearby centres, food at a named centre, or directions to one.")
    target_centre_id: Optional[str] = Field(
        default=None,
        description="Leave null; the application resolves named destinations against its verified centre catalogue.",
    )
    origin_place: Optional[str] = Field(
        default=None,
        description="A Singapore place explicitly named as the starting point, or null.",
    )
    travel_mode: Optional[Literal["pt", "walk", "drive", "cycle"]] = Field(
        default=None,
        description="Explicit travel mode only: pt, walk, drive, or cycle; otherwise null.",
    )
    proximity_preference: str = Field(
        description="Interpretation of proximity wording, e.g. closest, convenient, or flexible."
    )
    inferred_priorities: List[str] = Field(default_factory=list)
    interpretation: str = Field(description="Short explanation of the request's logistics intent.")


class CentreRanking(BaseModel):
    selected_centres: List[RankedCentre]
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    limitations: List[str] = Field(default_factory=list)


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the location-intent interpreter for SmartHawker.
Read the current user message and recent context. Distinguish the diner’s origin
from a hawker centre they ask about. A named centre in “what’s good at Maxwell?”
is a destination, never the origin. Preserve a prior origin unless the current
message explicitly changes it. Extract origin_place only when a starting place is
explicitly stated (e.g. “near Yio Chu Kang”). Classify centre-specific food questions
as explore_centre; directions requests as directions_to_centre; general discovery as
find_nearby_centres. Leave target_centre_id null; the application matches names to its
verified catalogue. If there is no origin, the caller may use browser coordinates.
Detect explicit transport mode only; public transport is the caller's default.
Interpret nuance such as 'near', 'fastest', 'easy to get to', 'avoid long walks', and
'few transfers' as preferences, not as facts. Never invent a precise radius or time limit."""),
    (
        "human",
        "Current message: {request}\nRecent turns: {recent_turns}\n"
        "Form values: {form_values}",
    ),
])

RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the SmartHawker Location & Logistics agent. Rank only centre IDs
present in the supplied evidence. Ground every claim in that evidence. Prefer the
fastest measured route when the user asks for fastest/most efficient. Otherwise weigh
the user's wording and priorities against measured route duration, walking time, and
transfer count. Do not invent transfers, accessibility, weather, or route details. If
route evidence is missing, say it is unavailable and lower confidence. Use an
alternative route only when that route is included in the evidence. Return a useful
shortlist, usually up to three centres. For walking, use the supplied rounded distance
and walking-time labels; never expose raw metre decimals in the rationale."""),
    ("human", "User request: {request}\nIntent interpretation: {intent}\nEvidence: {evidence}"),
])


def _llm(model_name: str) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for Location & Logistics reasoning.")
    return ChatOpenAI(model=model_name, temperature=0)


def _normalise_query(value: Any) -> LocationQuery:
    if isinstance(value, LocationQuery):
        return value
    if isinstance(value, str):
        return LocationQuery(request=value)
    payload = dict(value or {})
    if "request" not in payload:
        payload["request"] = payload.get("user_prompt", payload.get("task", ""))
    if "user_location" not in payload:
        lat = payload.pop("user_lat", payload.pop("lat", None))
        lng = payload.pop("user_lng", payload.pop("lng", None))
        if lat is not None and lng is not None:
            payload["user_location"] = {"lat": lat, "lng": lng}
    return LocationQuery.model_validate(payload)


def _explicit_centre_target(request: str, catalogue: List[Dict[str, Any]]) -> Optional[str]:
    """Match a centre the user names directly; never infer it from model prose."""
    def normalise(value: str) -> str:
        value = value.casefold().replace("&", " and ")
        value = re.sub(r"\bcenter\b", "centre", value)
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    text = f" {normalise(request)} "
    direct_matches: set[str] = set()
    partial_scores: Dict[str, int] = {}
    for centre in catalogue:
        name = centre["name"]
        parenthetical = re.findall(r"\(([^)]*)\)", name)
        aliases = {name, *parenthetical, *centre.get("aliases", [])}
        for alias in list(aliases):
            # The official NEA label may include a block/address before the
            # familiar centre name. Keep the parenthetical centre name as a
            # standalone exact alias and accept spelling/punctuation variants.
            cleaned = re.sub(
                r"\b(market\s+and\s+food\s+centre|market\s+food\s+centre|food\s+centre)$",
                "",
                alias,
                flags=re.IGNORECASE,
            ).strip(" ()")
            if cleaned and len(cleaned.split()) > 1:
                aliases.add(cleaned)
        normalised_aliases = {normalise(alias) for alias in aliases if alias}
        direct_match = any(alias and f" {alias} " in text for alias in normalised_aliases)
        # Support a uniquely named avenue/estate fragment such as "Ang Mo
        # Kio Ave 6" when it identifies one NEA centre. Short fragments such
        # as "Ang Mo Kio" remain ambiguous and are not guessed.
        if not direct_match:
            best_score = 0
            for alias in normalised_aliases:
                words = alias.split()
                if len(words) < 4:
                    continue
                for size in range(4, min(7, len(words) + 1)):
                    if any(
                        f" {' '.join(words[index:index + size])} " in text
                        for index in range(len(words) - size + 1)
                    ):
                        best_score = max(best_score, size)
            if best_score:
                partial_scores[centre["id"]] = best_score
        else:
            direct_matches.add(centre["id"])
    if len(direct_matches) == 1:
        return next(iter(direct_matches))
    if direct_matches:
        return None
    if not partial_scores:
        return None
    best_score = max(partial_scores.values())
    best_matches = [
        centre_id for centre_id, score in partial_scores.items()
        if score == best_score
    ]
    return best_matches[0] if len(best_matches) == 1 else None


def _canonical_origin(place: str) -> tuple[str, bool]:
    """Expand common Singapore place aliases and indicate specific landmarks."""
    cleaned = re.sub(r"\s+mrt\s+station$", "", place.strip(), flags=re.IGNORECASE)
    aliases = {
        "ngee ann poly": "Ngee Ann Polytechnic",
        "ngee ann polytechnic": "Ngee Ann Polytechnic",
    }
    canonical = aliases.get(cleaned.casefold(), cleaned)
    specific_suffixes = (
        "mrt", "station", "road", "street", "avenue", "building",
        "mall", "school", "poly", "polytechnic", "university", "hospital",
        "airport", "terminal", "hotel", "hub", "plaza", "complex", "campus",
        "food centre", "food center",
    )
    is_specific = canonical.casefold().endswith(specific_suffixes)
    return canonical, is_specific


def _resolve_origin(
    query: LocationQuery, intent: LocationIntent
) -> tuple[Dict[str, Any], str]:
    # Intent extracted from the current user message takes precedence over browser GPS.
    session_state = (query.conversation_context or {}).get("state", {})
    carried_place = session_state.get("location_text")
    if not carried_place and session_state.get("origin_source") == "free_text_place":
        carried_place = session_state.get("origin_label")
    # Structured model output is not ground truth: reject a supposed origin
    # unless the user actually wrote it this turn. Otherwise a hallucinated
    # place name can silently become the route origin.
    current_text = query.request.casefold()
    prior_user_text = " ".join(
        str(turn.get("content", ""))
        for turn in (query.conversation_context or {}).get("turns", [])
        if isinstance(turn, dict) and turn.get("role") == "user"
    ).casefold()
    model_place = intent.origin_place
    if model_place and not any(
        model_place.casefold() in source
        for source in (current_text, prior_user_text)
    ):
        model_place = None
    explicit_place = model_place or query.location_text or carried_place
    if explicit_place:
        canonical_place, is_specific = _canonical_origin(explicit_place)
        lookup = canonical_place
        # Neighbourhood-sized requests need a disclosed, repeatable anchor.
        # Prefer the local MRT station unless the user supplied a more specific
        # landmark, address, or facility.
        if not is_specific:
            lookup = f"{canonical_place} MRT Station"
        resolved_query = lookup
        resolved = geocode_place(lookup)
        if resolved is None and lookup != canonical_place:
            resolved = geocode_place(canonical_place)
            resolved_query = canonical_place
        if resolved is None:
            raise LocationResolutionError(
                f"I couldn't resolve '{canonical_place}' to a Singapore location. "
                "Try a nearby landmark or MRT station."
            )
        # Show the landmark query used for the coordinates, rather than an
        # opaque or overly broad address label returned by the search index.
        resolved["label"] = resolved_query
        return resolved, "free_text_place"
    if query.user_location is not None:
        source = query.user_location_source or "provided_coordinates"
        point = {
            "lat": query.user_location.lat,
            "lng": query.user_location.lng,
        }
        if not is_singapore_location(point):
            raise LocationResolutionError(
                "That starting point appears to be outside Singapore. "
                "Please provide a location within Singapore."
            )
        return {
            **point,
            "label": (
                "your current location"
                if source == "browser_current_location"
                else "provided coordinates"
            ),
            "source": source,
        }, source
    raise LocationResolutionError(
        "I need a starting point to compare nearby hawker centres. Allow location "
        "access or name a place, for example 'near Yio Chu Kang'."
    )


def _collect_logistics(
    query: LocationQuery,
    origin: Dict[str, Any],
    *,
    allow_short_walk_fallback: bool,
) -> List[CentreLogistics]:
    all_centres = get_candidate_centres()
    start = {"lat": origin["lat"], "lng": origin["lng"]}

    # Coordinates are used only to avoid paying for routes to obviously
    # irrelevant parts of Singapore. They are never returned as travel
    # distance or used as a substitute for OneMap route evidence.
    requested_area = (
        (query.parsed_request or {}).get("requested_area")
        or _requested_area(query.request)
    )
    area_anchors = {
        # Approximate geographic anchors, used only to select a broad candidate
        # pool when the user explicitly asks for a region.
        "west": (1.335, 103.705),
        "east": (1.350, 103.955),
        "north": (1.435, 103.790),
        "north_east": (1.370, 103.895),
        "south": (1.265, 103.820),
        "central": (1.300, 103.840),
    }
    search_point = area_anchors.get(requested_area, (start["lat"], start["lng"]))
    candidate_limit = max(1, int(os.getenv("LOCATION_ROUTE_CANDIDATE_LIMIT", "30")))
    all_centres = sorted(
        all_centres,
        key=lambda centre: _haversine_km(
            search_point[0], search_point[1],
            float(centre["latitude"]), float(centre["longitude"]),
        ),
    )[:candidate_limit]

    def evaluate(centre: Dict[str, Any]) -> CentreLogistics:
        limitations = []
        destination = {"lat": centre["latitude"], "lng": centre["longitude"]}
        route_data = None
        try:
            route_data = get_route(start, destination, query.travel_mode or "pt")
        except LocationServiceError as error:
            route_data = None
            limitations.append(str(error))
        # Public transport remains the default. Only when the user did not
        # choose a mode, use an actual walking itinerary if it is very short.
        alternative_route = None
        if allow_short_walk_fallback and query.travel_mode == "pt":
            try:
                walking_route = get_route(start, destination, "walk")
                threshold = int(os.getenv("LOCATION_SHORT_WALK_THRESHOLD_MIN", "8"))
                transit_minutes = (
                    route_data.get("total_duration_min") if route_data else None
                )
                if (
                    walking_route["total_duration_min"] <= threshold
                    and (
                        transit_minutes is None
                        or walking_route["total_duration_min"] <= transit_minutes
                    )
                ):
                    alternative_route = route_data
                    route_data = walking_route
                    limitations.append(
                        "Walking is suggested because its measured route is short "
                        "and no slower than the public-transport route."
                    )
                elif walking_route["total_duration_min"] <= threshold:
                    alternative_route = walking_route
            except LocationServiceError as error:
                limitations.append(
                    "Walking route could not be checked for the short-distance "
                    f"exception: {error}"
                )
        route = TravelEstimate.model_validate(route_data) if route_data else TravelEstimate(
            total_duration_min=None,
            mode=query.travel_mode or "pt",
            source="unavailable",
            confidence=0.0,
            limitation="No route duration was returned by OneMap.",
        )
        return CentreLogistics(
            centre_id=centre["id"],
            centre_name=centre["name"],
            address=centre["address"],
            coordinates={"lat": centre["latitude"], "lng": centre["longitude"]},
            travel=route,
            alternative_travel=(
                TravelEstimate.model_validate(alternative_route)
                if alternative_route else None
            ),
            evidence=[
                "Centre is in the NEA hawker-centre catalogue",
                f"Coordinates from {centre.get('coordinate_source', 'NEA geospatial data')}",
            ],
            limitations=limitations,
        )

    if not all_centres:
        return []
    worker_count = max(1, min(
        len(all_centres), int(os.getenv("LOCATION_ROUTE_WORKERS", "6"))
    ))
    # Keep the catalogue order while routing independently in a small pool.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(evaluate, all_centres))


def _requested_area(request: str) -> Optional[str]:
    """Return an explicitly requested Singapore region, if present."""
    text = request.casefold()
    patterns = {
        "west": r"\b(?:west|western)\b",
        "east": r"\b(?:east|eastern)\b",
        "north": r"\b(?:north|northern)\b",
        "south": r"\b(?:south|southern)\b",
        "central": r"\b(?:central|city centre|city center)\b",
    }
    matches = [
        (match.start(), area)
        for area, pattern in patterns.items()
        if (match := re.search(pattern, text))
    ]
    return max(matches)[1] if matches else None


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approximate geographic separation for internal candidate filtering only."""
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(value))


def _shortlist_for_reasoning(
    query: LocationQuery,
    intent: LocationIntent,
    logistics: List[CentreLogistics],
    limit: int = 8,
) -> List[CentreLogistics]:
    """Send only the strongest route candidates to the ranking LLM."""
    text = " ".join([
        query.request.casefold(),
        " ".join(query.priorities).casefold(),
        " ".join(intent.inferred_priorities).casefold(),
        intent.proximity_preference.casefold(),
    ])
    wants_least_walking = any(term in text for term in (
        "walk", "walking", "avoid long walks", "less walking",
    ))
    wants_fewer_transfers = any(term in text for term in (
        "transfer", "few transfers", "no transfers",
    ))
    routed = [item for item in logistics if item.travel.total_duration_min is not None]
    if wants_least_walking:
        key = lambda item: (
            item.travel.walking_distance_m
            if item.travel.walking_distance_m is not None else math.inf,
            item.travel.total_duration_min,
        )
    elif wants_fewer_transfers:
        key = lambda item: (
            item.travel.transfers if item.travel.transfers is not None else math.inf,
            item.travel.total_duration_min,
        )
    else:
        key = lambda item: (
            item.travel.total_duration_min,
            item.travel.walking_distance_m
            if item.travel.walking_distance_m is not None else math.inf,
        )
    return sorted(routed, key=key)[:limit]


def _rank_centres(
    query: LocationQuery,
    intent: LocationIntent,
    logistics: List[CentreLogistics],
) -> CentreRanking:
    if not logistics:
        return CentreRanking(
            selected_centres=[],
            reasoning="No hawker centre could be loaded from the NEA catalogue.",
            confidence=0.0,
            limitations=["No usable centre coordinate evidence was returned."],
        )
    eligible = [
        item for item in logistics
        if query.max_travel_time_min is None
        or (
            item.travel.total_duration_min is not None
            and item.travel.total_duration_min <= query.max_travel_time_min
        )
    ]
    if not eligible:
        return CentreRanking(
            selected_centres=[],
            reasoning="No centre has measured route evidence within the stated travel-time limit.",
            confidence=0.8,
            limitations=[
                "A missing route duration cannot pass the travel-time limit."
            ],
        )
    shortlist = _shortlist_for_reasoning(query, intent, eligible)
    if not shortlist:
        return CentreRanking(
            selected_centres=[],
            reasoning="OneMap did not return a usable route for the supported centres.",
            confidence=0.0,
            limitations=["No centre can be ranked by actual travel time without route evidence."],
        )
    llm = _llm(os.getenv("LOCATION_AGENT_MODEL", "gpt-4o"))
    chain = RANK_PROMPT | llm.with_structured_output(
        CentreRanking, method="function_calling"
    )
    allowed = {item.centre_id for item in shortlist}
    result: CentreRanking = chain.invoke({
        "request": query.request,
        "intent": {
            **intent.model_dump(),
            "travel_mode": query.travel_mode,
            "max_travel_time_min": query.max_travel_time_min,
            "priorities": query.priorities,
        },
        "evidence": json.dumps([
            {
                "centre_id": item.centre_id,
                "centre_name": item.centre_name,
                "travel": {
                    "minutes": item.travel.total_duration_min,
                    "mode": item.travel.mode,
                    "walking_distance": item.travel.walking_distance_display,
                    "walking_time": item.travel.walking_time_display,
                    "transfers": item.travel.transfers,
                    "source": item.travel.source,
                },
                "alternative_travel": (
                    {
                        "minutes": item.alternative_travel.total_duration_min,
                        "mode": item.alternative_travel.mode,
                        "walking_distance": item.alternative_travel.walking_distance_display,
                        "walking_time": item.alternative_travel.walking_time_display,
                    }
                    if item.alternative_travel else None
                ),
            }
            for item in shortlist
        ], ensure_ascii=False),
    })
    # Guardrail against invented IDs, duplicate ranks, and centres excluded by
    # a hard travel-time constraint.
    unique = []
    seen = set()
    for selection in sorted(result.selected_centres, key=lambda item: item.rank):
        if selection.centre_id not in allowed or selection.centre_id in seen:
            continue
        seen.add(selection.centre_id)
        unique.append(selection.model_copy(update={"rank": len(unique) + 1}))
    if not unique:
        fastest = min(shortlist, key=lambda item: item.travel.total_duration_min)
        unique = [RankedCentre(
            centre_id=fastest.centre_id,
            rank=1,
            logistics_fit=0.5,
            rationale="Fastest verified fallback among the shortlisted route candidates.",
        )]
        result.limitations.append(
            "The model returned no valid centre IDs; a verified fallback was used."
        )
    return result.model_copy(update={"selected_centres": unique[:3]})


def run(value: Any) -> Dict[str, Any]:
    """Interpret a location request, gather route evidence, and rank centres."""
    query = _normalise_query(value)
    state = (query.conversation_context or {}).get("state", {})
    parsed = query.parsed_request
    if parsed:
        # The orchestrator has already extracted the shared request fields.
        # Keep this agent's own route reasoning, but don't parse the prompt a
        # second time and risk conflicting interpretations.
        intent_names = {
            "food_discovery": "find_nearby_centres",
            "centre_details": "explore_centre",
            "directions": "directions_to_centre",
        }
        parsed_origin = parsed.get("origin_text")
        user_text = " ".join(
            str(turn.get("content", ""))
            for turn in (query.conversation_context or {}).get("turns", [])
            if isinstance(turn, dict) and turn.get("role") == "user"
        )
        if parsed_origin and any(
            parsed_origin.casefold() in source.casefold()
            for source in (query.request, user_text)
        ):
            query = query.model_copy(update={"location_text": parsed_origin})
        query = query.model_copy(update={
            "travel_mode": query.travel_mode or parsed.get("travel_mode"),
            "max_travel_time_min": (
                query.max_travel_time_min
                if query.max_travel_time_min is not None
                else parsed.get("max_travel_time_min")
            ),
            "priorities": query.priorities or parsed.get("priorities", []),
        })
        intent = LocationIntent(
            request_intent=intent_names.get(
                parsed.get("intent"), "find_nearby_centres"
            ),
            target_centre_id=None,
            origin_place=parsed_origin,
            travel_mode=parsed.get("travel_mode"),
            proximity_preference=(
                ", ".join(parsed.get("priorities", [])) or "convenient"
            ),
            inferred_priorities=parsed.get("priorities", []),
            interpretation=parsed.get("clarification_question") or parsed.get("intent", ""),
        )
    else:
        # Compatibility for direct /api/location calls that haven't come
        # through the orchestrator.
        model = _llm(os.getenv("LOCATION_AGENT_MODEL", "gpt-4o"))
        intent_chain = INTENT_PROMPT | model.with_structured_output(
            LocationIntent, method="function_calling"
        )
        raw_turns = (query.conversation_context or {}).get("turns", [])[-4:]
        recent_turns = [
            {
                "role": turn.get("role"),
                "content": str(turn.get("content", ""))[:400],
            }
            for turn in raw_turns
            if isinstance(turn, dict)
        ]
        form_values = {
            "travel_mode": query.travel_mode,
            "max_travel_time_min": query.max_travel_time_min,
            "priorities": query.priorities,
            "session_state": state,
        }
        intent = intent_chain.invoke({
            "request": query.request,
            "recent_turns": json.dumps(recent_turns, ensure_ascii=False),
            "form_values": json.dumps(form_values, ensure_ascii=False),
        })
    catalogue = load_hawker_centres()
    valid_centres = {centre["id"]: centre for centre in catalogue}
    explicit_target = _explicit_centre_target(query.request, catalogue)
    if explicit_target is None and parsed:
        explicit_target = _explicit_centre_target(
            str(parsed.get("target_centre_text") or ""), catalogue
        )
    requested_target = str((parsed or {}).get("target_centre_text") or "").strip()
    if not requested_target:
        # Direct location API calls may not include an orchestrator parse.
        # Still treat an explicit named-destination question as a target.
        target_match = re.search(
            r"\b(?:at|in|inside|to|get to|about)\s+(.+?)(?:\?|$)",
            query.request,
            flags=re.IGNORECASE,
        )
        if target_match and re.search(
            r"\b(?:what|options?|food|good|directions?|how to get|route)\b",
            query.request,
            flags=re.IGNORECASE,
        ):
            requested_target = target_match.group(1).strip(" .")
    if requested_target and not explicit_target:
        # A requested destination that is absent from the verified centre
        # catalogue must never degrade into a search around the previous origin.
        return LocationDecision(
            selected_centres=[],
            logistics=[],
            reasoning=(
                f"The named destination '{requested_target}' could not be matched "
                "to a centre in the verified NEA catalogue."
            ),
            confidence=0.0,
            limitations=[
                "No centre or menu data was returned because the named destination could not be verified."
            ],
            intent_interpretation={
                **intent.model_dump(),
                "target_centre_text": requested_target,
                "target_resolution": "not_found",
            },
        ).model_dump(mode="json")
    if explicit_target:
        lowered_request = query.request.casefold()
        is_directions = bool(re.search(
            r"\b(how to get|directions?|route|travel|go to|get to|way to)\b",
            lowered_request,
        ))
        intent = intent.model_copy(update={
            "target_centre_id": explicit_target,
            "request_intent": "directions_to_centre" if is_directions else "explore_centre",
        })
    if intent.target_centre_id not in valid_centres:
        intent = intent.model_copy(update={"target_centre_id": None})
    if intent.request_intent in {"explore_centre", "directions_to_centre"} and intent.target_centre_id:
        # A centre-specific follow-up is a different task from nearby-centre
        # discovery. Return only the grounded centre; the chat layer can route
        # stall/dish questions to a catalogue-backed recommendation worker.
        centre = valid_centres[intent.target_centre_id]
        if intent.request_intent == "explore_centre":
            return LocationDecision(
                selected_centres=[RankedCentre(
                    centre_id=centre["id"], rank=1, logistics_fit=1.0,
                    rationale="This is the centre named in your current message.",
                )],
                logistics=[],
                reasoning=(
                    f"You asked about {centre['name']}; this turn is centre-specific, "
                    "not a nearby-centre search."
                ),
                confidence=1.0,
                limitations=[],
                intent_interpretation={
                    **intent.model_dump(),
                    "target_centre_name": centre["name"],
                    "travel_mode": query.travel_mode or state.get("travel_mode") or "pt",
                    "origin_preserved": state.get("origin_label"),
                },
            ).model_dump(mode="json")
    # Preserve whether the user chose a mode before applying the PT default.
    requested_mode = query.travel_mode or intent.travel_mode or state.get("travel_mode")
    allow_short_walk_fallback = requested_mode is None
    selected_mode = requested_mode if requested_mode in {"pt", "walk", "drive", "cycle"} else "pt"
    query = query.model_copy(update={"travel_mode": selected_mode})
    origin, origin_source = _resolve_origin(query, intent)
    if intent.request_intent == "directions_to_centre" and intent.target_centre_id:
        target = valid_centres[intent.target_centre_id]
        candidates = get_candidate_centres()
        candidate = next((c for c in candidates if c["id"] == target["id"]), None)
        if candidate is None:
            raise LocationResolutionError(f"I couldn't locate {target['name']} in the NEA catalogue.")
        route_data = get_route(
            {"lat": origin["lat"], "lng": origin["lng"]},
            {"lat": candidate["latitude"], "lng": candidate["longitude"]},
            selected_mode,
        )
        travel = TravelEstimate.model_validate(route_data)
        logistics = [CentreLogistics(
            centre_id=target["id"], centre_name=target["name"],
            address=target["address"],
            coordinates={"lat": candidate["latitude"], "lng": candidate["longitude"]},
            travel=travel,
            evidence=["Destination is in the NEA hawker-centre catalogue", "Route returned by OneMap"],
        )]
        return LocationDecision(
            selected_centres=[RankedCentre(
                centre_id=target["id"], rank=1, logistics_fit=1.0,
                rationale="The route is to the centre named in your message.",
            )],
            logistics=logistics,
            reasoning=f"Route from {origin.get('label') or 'your stated origin'} to {target['name']}.",
            confidence=travel.confidence,
            origin={"lat": origin["lat"], "lng": origin["lng"]},
            origin_label=origin.get("label"),
            origin_source=origin_source,
            intent_interpretation={
                **intent.model_dump(), "target_centre_name": target["name"],
                "travel_mode": selected_mode,
            },
        ).model_dump(mode="json")
    logistics = _collect_logistics(
        query,
        origin,
        allow_short_walk_fallback=allow_short_walk_fallback,
    )
    ranking = _rank_centres(query, intent, logistics)
    catalogue_ids = {centre["id"] for centre in catalogue}
    # The ranking model is restricted to catalogue IDs. This check protects
    # downstream workers consuming this response from model-generated centres.
    ranking.selected_centres = [
        item for item in ranking.selected_centres if item.centre_id in catalogue_ids
    ]
    limitations = list(ranking.limitations)
    if not any(item.travel.source == "onemap" for item in logistics):
        limitations.append(
            "Route durations are unavailable, so centre travel convenience could not be compared."
        )
    return LocationDecision(
        selected_centres=ranking.selected_centres,
        logistics=logistics,
        reasoning=ranking.reasoning,
        confidence=ranking.confidence,
        limitations=limitations,
        origin={"lat": origin["lat"], "lng": origin["lng"]},
        origin_label=origin.get("label"),
        origin_source=origin_source,
        intent_interpretation={
            **intent.model_dump(),
            "travel_mode": selected_mode,
            "short_walk_fallback_enabled": allow_short_walk_fallback,
            "explicit_location_precedence": "free_text_place_over_browser_coordinates",
        },
    ).model_dump(mode="json")
