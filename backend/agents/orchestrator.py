"""Parse requests, route location first, run relevant workers, and verify outputs."""

import json
import operator
import os
import re
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

from models.orchestration import (
    AgentResult,
    OrchestrationPlan,
    ParsedRequest,
    PlannedTask,
)


class SharedState(TypedDict, total=False):
    user_request: str
    conversation_context: dict
    user_location: Optional[dict]
    user_location_source: Optional[str]
    model: ChatOpenAI
    parsed_request: dict
    tasks: list[PlannedTask]
    location_result: Optional[dict]
    agent_results: Annotated[list[dict], operator.add]
    verified_results: list[dict]
    final_recommendation: str
    needs_location: bool
    location_reused: bool


class WorkerState(TypedDict, total=False):
    task: PlannedTask
    user_request: str
    conversation_context: dict
    parsed_request: dict
    user_location: Optional[dict]
    user_location_source: Optional[str]
    location_result: Optional[dict]
    excluded_menu_item_ids: list[str]
    agent_results: Annotated[list[dict], operator.add]


def build_model(state: SharedState) -> SharedState:
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to backend/.env or export it before starting the backend."
        )
    state["model"] = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    return state


def parse_request(state: SharedState) -> SharedState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You parse SmartHawker requests and create a compact worker plan.
Use only information in the current message and recent context. The current message
is the active request; recent turns resolve references and preserve prior preferences.
Do not invent an origin, budget, time limit, dietary restriction, or transport mode.
`request.intent` describes the overall user goal, not a worker name. For queue or
crowd requests, use `food_discovery` as the intent and include a `queue` worker task.
Do not ask a confirmation question that merely repeats a clear request (for example,
"Are you looking for chicken rice near AMK Hub?"). Treat a named place as usable
origin context. Ask only when the request cannot be acted on after using recent turns;
the location worker handles a missing origin and the frontend may then offer GPS.
Extract specifically requested dishes or menu items into `menu_item_preferences`;
extract stall names only when the user explicitly names them into `stall_name_preferences`;
leave both lists empty for broad requests such as “what's good here?”.
For every populated field, record whether it came from the current message,
conversation context, form input, or inference in `field_sources`; list ambiguity
in `uncertain_fields` and ask a concise clarification when needed.
`origin_text` is a place the diner starts from; an area requested as a destination
goes in `requested_area`. If a follow-up refers to a previously named hawker centre,
put its name in `target_centre_text`. If the user asks for “other”, “different”, or
“another” option, exclude menu-item IDs previously recommended in session state.
If no origin is stated, leave it null so the app can use browser coordinates. Use SGD
for a stated dollar budget. Select Location when new route evidence or a newly named
centre needs resolving; otherwise reuse the prior shortlist in session state. Select
other workers only when their information is relevant. Each task should state the
specific work and constraints.
For greetings or unrelated requests, use intent `smalltalk` or `other` and no tasks."""),
        ("human", "Current message: {message}\nRecent context: {context}"),
    ])
    parser = prompt | state["model"].with_structured_output(
        OrchestrationPlan, method="function_calling"
    )
    context = state.get("conversation_context", {})
    plan: OrchestrationPlan = parser.invoke({
        "message": state.get("user_request", ""),
        "context": json.dumps(context, ensure_ascii=False),
    })
    request, queue_intent = _normalize_queue_intent(plan.request)
    allowed_sources = {
        "current_message", "conversation_context", "form_input", "inferred"
    }
    request.field_sources = {
        key: (
            value.strip().casefold().replace(" ", "_")
            if value.strip().casefold().replace(" ", "_") in allowed_sources
            else "inferred"
        )
        for key, value in request.field_sources.items()
    }
    session_state = (context or {}).get("state", {})
    current_message = state.get("user_request", "")
    if not request.origin_text:
        explicit_origin = _explicit_origin_from_message(current_message)
        if explicit_origin:
            request.origin_text = explicit_origin
            request.field_sources["origin_text"] = "current_message"
    if not request.target_centre_text and re.search(
        r"\b(at|in|inside)\b", current_message, re.I
    ):
        from agents.location_agent import _explicit_centre_target
        from services.location_service import load_hawker_centres

        catalogue = load_hawker_centres()
        centre_id = _explicit_centre_target(current_message, catalogue)
        if centre_id:
            request.target_centre_text = next(
                centre["name"] for centre in catalogue if centre["id"] == centre_id
            )
    if re.search(r"\b(other|different|another)\b", current_message, re.I):
        prior_ids = session_state.get("previous_menu_item_ids", [])
        if prior_ids:
            request.excluded_menu_item_ids = list(dict.fromkeys(prior_ids))
        if re.search(r"\bother\s+food\s+options?\b", current_message, re.I):
            request.menu_item_preferences = []
            request.stall_name_preferences = []
            request.cuisine_preferences = []
    # Prevent the planner from turning a complete dish + place request into
    # a needless yes/no loop. If the request names what to find and where to
    # start, location resolution can proceed or ask for the genuinely missing
    # origin itself.
    if (
        request.clarification_question
        and request.intent == "food_discovery"
        and (request.menu_item_preferences or request.cuisine_preferences)
        and (request.origin_text or request.requested_area)
    ):
        request.clarification_question = None
    tasks = _normalize_tasks(
        plan.tasks, request, current_message, context,
        queue_requested=queue_intent,
    )
    return {
        "parsed_request": request.model_dump(),
        "tasks": tasks,
    }


def _normalize_queue_intent(request: ParsedRequest) -> tuple[ParsedRequest, bool]:
    if request.intent != "queue":
        return request, False
    return request.model_copy(update={"intent": "food_discovery"}), True


def _requests_queue_data(message: str) -> bool:
    return bool(re.search(
        r"\b(?:queues?|crowds?|crowded|waiting\s+times?|wait\s+times?)\b",
        message,
        re.IGNORECASE,
    ))


def _explicit_origin_from_message(message: str) -> Optional[str]:
    match = re.search(
        r"\b(?:near|around|from)\s+(.+?)"
        r"(?=\s+(?:for|with|under|within|and|but|while)\b|[,;.!?]|$)",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    place = match.group(1).strip(" \t\r\n,.;:!?\"'")
    return place or None


def _stall_name_terms(value: str) -> set[str]:
    generic_terms = {
        "demo", "stall", "hawker", "centre", "center", "food",
        "queue", "queues", "crowd", "crowds", "the", "for", "at",
        "and", "i", "want", "to", "know", "is", "what", "about",
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return {
        term for term in normalized.split()
        if term not in generic_terms and not term.isdigit()
    }


def _normalize_tasks(
    tasks: list[PlannedTask], parsed: ParsedRequest, message: str,
    conversation_context: Optional[dict] = None,
    *, queue_requested: bool = False,
) -> list[PlannedTask]:
    """Enforce required worker calls from validated constraints and intent."""
    queue_requested = queue_requested or _requests_queue_data(message)
    by_agent = {task.agent: task for task in tasks}
    session_state = (conversation_context or {}).get("state", {})
    cached_location = session_state.get("location_result") or {}
    has_cached_location = bool(cached_location.get("selected_centres"))
    cached_origin = str(cached_location.get("origin_label") or "").casefold()
    requested_origin = str(parsed.origin_text or "").strip().casefold()
    origin_changed = bool(
        requested_origin
        and requested_origin in message.casefold()
        and requested_origin != cached_origin
    )
    route_preference_changed = bool(
        parsed.travel_mode
        and parsed.travel_mode != session_state.get("travel_mode")
    ) or (
        parsed.max_travel_time_min is not None
        and parsed.max_travel_time_min != session_state.get("max_travel_time_min")
    ) or bool(
        parsed.priorities
        and parsed.priorities != session_state.get("priorities", [])
    )
    needs_location = (
        parsed.intent == "directions"
        or bool(parsed.target_centre_text)
        or (
            parsed.intent in {"food_discovery", "centre_details"}
            and (
                not has_cached_location
                or origin_changed
                or route_preference_changed
                or bool(parsed.requested_area)
            )
        )
    )
    if has_cached_location and not needs_location:
        by_agent.pop("location", None)
    if needs_location and "location" not in by_agent:
        by_agent["location"] = PlannedTask(
            agent="location",
            task=(
                "Resolve the named centre locally; request routes only if the user "
                "asks for journey evidence."
                if parsed.target_centre_text else
                "Resolve the changed origin or requested area and gather route evidence."
            ),
        )
    if parsed.intent in {"food_discovery", "centre_details"} and "dietary" not in by_agent:
        by_agent["dietary"] = PlannedTask(
            agent="dietary", task="Retrieve stall candidates, applying any dietary or cuisine constraints."
        )
    if parsed.dietary_preferences and "dietary" not in by_agent:
        by_agent["dietary"] = PlannedTask(
            agent="dietary", task="Check the stated dietary constraints against stall data."
        )
    if parsed.cuisine_preferences and "dietary" not in by_agent:
        by_agent["dietary"] = PlannedTask(
            agent="dietary", task="Match the stated cuisine preferences against stall data."
        )
    if parsed.budget_amount is not None and "budget" not in by_agent:
        by_agent["budget"] = PlannedTask(
            agent="budget", task="Filter stalls by the user's stated budget."
        )
    if (queue_requested or parsed.max_queue_min is not None) and "queue" not in by_agent:
        by_agent["queue"] = PlannedTask(
            agent="queue",
            task=(
                "Assess mock queue and crowd evidence for the user's request, "
                "applying any stated time period and queue limit."
            ),
        )
    weather_terms = ("weather", "rain", "rainy", "outdoor", "wet")
    if any(term in message.casefold() for term in weather_terms) and "weather" not in by_agent:
        by_agent["weather"] = PlannedTask(
            agent="weather", task="Assess whether current weather affects the options."
        )
    if parsed.intent in {"smalltalk", "other"}:
        return []
    ordered = []
    if needs_location and "location" in by_agent:
        ordered.append(by_agent.pop("location"))
    ordered.extend(by_agent.values())
    return ordered[:5]


def route_after_parse(state: SharedState) -> str:
    if state.get("parsed_request", {}).get("clarification_question"):
        return "clarification_response"
    if not state.get("tasks"):
        return "respond_without_workers"
    if any(task.agent == "location" for task in state.get("tasks", [])):
        return "location_worker"
    if (state.get("conversation_context", {}).get("state", {}).get("location_result") or {}).get("selected_centres"):
        return "reuse_location"
    return "location_worker"


def _result_envelope(agent: str, payload: dict[str, Any], **updates: Any) -> dict[str, Any]:
    status = updates.pop("status", payload.get("status", "ok"))
    detail_payload = {
        key: value for key, value in payload.items()
        if key not in {"candidates", "evidence", "status"}
    }
    return AgentResult(
        agent=agent,
        status=status,
        candidates=payload.get("candidates", []),
        evidence=payload.get("evidence", []),
        confidence=payload.get("confidence", 0.5),
        limitations=payload.get("limitations", []),
        error=payload.get("error"),
        location_required=payload.get("location_required", False),
        payload=detail_payload,
        **updates,
    ).model_dump()


def location_worker(state: SharedState) -> dict[str, Any]:
    task = next((item for item in state["tasks"] if item.agent == "location"), None)
    if task is None:
        return {"location_result": None, "needs_location": False}
    from agents.location_agent import LocationResolutionError, run
    from services.location_service import LocationServiceError

    try:
        result = run({
            "request": state["user_request"],
            "task": task.task,
            "parsed_request": state["parsed_request"],
            "conversation_context": state.get("conversation_context", {}),
            "user_location": state.get("user_location"),
            "user_location_source": state.get("user_location_source"),
        })
        result["agent"] = "location"
        needs_location = False
    except LocationResolutionError as error:
        result = {
            "agent": "location",
            "location_required": True,
            "error": str(error),
        }
        needs_location = True
    except LocationServiceError as error:
        result = {
            "agent": "location",
            "error": str(error),
            "limitations": ["OneMap location or route evidence is unavailable."],
        }
        needs_location = False
    envelope = _result_envelope(
        "location", result,
        status=(
            "needs_input" if needs_location
            else "partial" if result.get("error")
            else "ok"
        ),
    )
    return {
        "location_result": result,
        "needs_location": needs_location,
        "agent_results": [envelope],
    }


def reuse_location(state: SharedState) -> dict[str, Any]:
    """Reuse the prior verified shortlist when this turn changes food filters only."""
    cached = (
        state.get("conversation_context", {})
        .get("state", {})
        .get("location_result")
    )
    if not isinstance(cached, dict) or not cached.get("selected_centres"):
        return {"location_result": None, "location_reused": False}
    return {
        "location_result": cached,
        "location_reused": True,
        "agent_results": [_result_envelope("location", cached)],
    }


def expand_menu_search(state: SharedState) -> dict[str, Any]:
    """If the nearest shortlist has no catalogue match, use further routed centres."""
    parsed = state.get("parsed_request", {})
    location = dict(state.get("location_result") or {})
    if (
        state.get("needs_location")
        or state.get("location_reused")
        or parsed.get("intent") != "food_discovery"
        or not location.get("selected_centres")
    ):
        return {}

    # Queue values are mock data and cannot safely drive a search expansion.
    # Keep the initial route shortlist in that case and let the response explain
    # that the queue constraint could not be verified.
    if parsed.get("max_queue_min") is not None:
        return {}

    from services.hawker_service import (
        find_dietary_stalls,
        find_stalls_within_budget,
    )

    routes = [
        item for item in location.get("logistics", [])
        if isinstance(item, dict)
        and item.get("centre_id")
        and isinstance(item.get("travel"), dict)
        and item["travel"].get("total_duration_min") is not None
    ]
    if not routes:
        return {}

    initial = location.get("selected_centres", [])
    initial_ids = {
        item.get("centre_id") for item in initial
        if isinstance(item, dict) and item.get("centre_id")
    }
    menu_preferences = parsed.get("menu_item_preferences", [])
    stall_preferences = parsed.get("stall_name_preferences", [])
    cuisine_preferences = parsed.get("cuisine_preferences", [])
    dietary_preferences = parsed.get("dietary_preferences", [])
    budget = parsed.get("budget_amount")

    # A broad "what food is available?" request does not need a 24-centre
    # dish search. The menu worker will return entries for the Location
    # shortlist, while synthesis can distinguish an uncovered catalogue from
    # a genuine no-match for a requested dish.
    if not any((menu_preferences, stall_preferences, cuisine_preferences,
                dietary_preferences, budget is not None)):
        return {}

    text = " ".join([
        state.get("user_request", "").casefold(),
        " ".join(parsed.get("priorities", [])).casefold(),
    ])
    willing_to_travel_farther = bool(re.search(
        r"\b(willing to travel farther|willing to travel further|"
        r"don't mind travelling farther|don't mind travelling further|"
        r"don't mind going farther|don't mind going further|"
        r"do not mind travelling farther|do not mind travelling further|"
        r"further away is fine|farther away is fine|can travel further|"
        r"can travel farther|anywhere in singapore|no travel limit)\b",
        text,
    ))
    hard_max_minutes = parsed.get("max_travel_time_min")
    default_horizon = max(1, int(os.getenv("LOCATION_DEFAULT_SEARCH_HORIZON_MIN", "30")))
    horizon_minutes = (
        hard_max_minutes if hard_max_minutes is not None
        else None if willing_to_travel_farther
        else default_horizon
    )

    route_by_id = {item["centre_id"]: item for item in routes}
    if any(term in text for term in ("least walking", "less walking", "shortest walk", "walking")):
        sort_key = lambda item: (
            item.get("travel", {}).get("walking_distance_m")
            if item.get("travel", {}).get("walking_distance_m") is not None else float("inf"),
            item["travel"]["total_duration_min"],
        )
    elif any(term in text for term in ("few transfers", "fewer transfers", "no transfers")):
        sort_key = lambda item: (
            item.get("travel", {}).get("transfers")
            if item.get("travel", {}).get("transfers") is not None else float("inf"),
            item["travel"]["total_duration_min"],
        )
    else:
        sort_key = lambda item: (
            item["travel"]["total_duration_min"],
            item.get("travel", {}).get("walking_distance_m")
            if item.get("travel", {}).get("walking_distance_m") is not None else float("inf"),
        )

    ordered_routes = sorted(routes, key=sort_key)
    within_horizon = [
        route for route in ordered_routes
        if horizon_minutes is None
        or route["travel"]["total_duration_min"] <= horizon_minutes
    ]
    farther_routes = [
        route for route in ordered_routes
        if horizon_minutes is not None
        and hard_max_minutes is None
        and route["travel"]["total_duration_min"] > horizon_minutes
    ]

    def matching_items(centre_id: str) -> list[dict[str, Any]]:
        matches = find_dietary_stalls(
            dietary_preferences=dietary_preferences,
            cuisine_preferences=cuisine_preferences,
            menu_item_preferences=menu_preferences,
            stall_name_preferences=stall_preferences,
            excluded_menu_item_ids=parsed.get("excluded_menu_item_ids", []),
            candidate_centre_ids=[centre_id],
        )
        if budget is not None:
            within_budget = {
                item.get("menu_item_id")
                for item in find_stalls_within_budget(
                    budget=budget,
                    candidate_centre_ids=[centre_id],
                    excluded_menu_item_ids=parsed.get("excluded_menu_item_ids", []),
                )
            }
            matches = [
                item for item in matches
                if item.get("menu_item_id") in within_budget
            ]
        return matches

    initial_summary = []
    for selected in initial:
        if isinstance(selected, dict) and selected.get("centre_id"):
            summary = route_by_id.get(selected["centre_id"])
            initial_summary.append({
                "centre_id": selected["centre_id"],
                "centre_name": (summary or {}).get("centre_name"),
                "travel_time_min": ((summary or {}).get("travel") or {}).get("total_duration_min"),
                "walking_distance_display": ((summary or {}).get("travel") or {}).get("walking_distance_display"),
                "walking_time_display": ((summary or {}).get("travel") or {}).get("walking_time_display"),
            })

    if dietary_preferences:
        location["selected_centres"] = [
            selected for selected in initial
            if isinstance(selected, dict)
            and selected.get("centre_id") in {r["centre_id"] for r in within_horizon}
        ]
        location["search_expansion"] = {
            "outcome": "dietary_verification_unavailable",
            "initial_centres": initial_summary,
            "search_horizon_min": horizon_minutes,
            "route_checked_centre_count": len(within_horizon),
            "note": "Menu data has no ingredient, allergen, or certification fields to verify this restriction.",
        }
        location.setdefault("limitations", []).append(
            "Nearby menu entries cannot be checked against the requested dietary restriction."
        )
        return {"location_result": location}

    matched_within = [
        (route, matching_items(route["centre_id"]))
        for route in within_horizon
    ]
    matched_within = [pair for pair in matched_within if pair[1]]
    matched_farther = []
    if not matched_within and hard_max_minutes is None and not willing_to_travel_farther:
        matched_farther = [
            (route, matching_items(route["centre_id"]))
            for route in farther_routes
        ]
        matched_farther = [pair for pair in matched_farther if pair[1]][:3]

    def centre_summary(centre_id: str) -> dict[str, Any]:
        route = route_by_id.get(centre_id, {})
        travel = route.get("travel") or {}
        return {
            "centre_id": centre_id,
            "centre_name": route.get("centre_name"),
            "travel_time_min": travel.get("total_duration_min"),
            "travel_mode": travel.get("mode"),
            "transfers": travel.get("transfers"),
            "walking_distance_display": travel.get("walking_distance_display"),
            "walking_time_display": travel.get("walking_time_display"),
        }

    selected_matches = matched_within[:3]
    if selected_matches:
        outcome = (
            "matched_in_initial_shortlist"
            if any(route["centre_id"] in initial_ids for route, _ in selected_matches)
            else "expanded_match_found"
        )
        selected_to_use = selected_matches
    else:
        # Keep the nearest in-horizon locations as useful context for the
        # clarification, even though they did not match the requested dish.
        selected_to_use = [
            (route, []) for route in within_horizon[:3]
        ]
        outcome = (
            "farther_match_requires_approval" if matched_farther
            else "no_match_in_routed_pool"
        )

    search_info = {
        "outcome": outcome,
        "trigger": "Checked the menu constraints against route-ranked centres and catalogue coverage.",
        "initial_centres": initial_summary,
        "search_horizon_min": horizon_minutes,
        "route_checked_centre_count": len(within_horizon),
        "route_candidate_pool_size": len(routes),
        "route_time_min_range": (
            [
                min(route["travel"]["total_duration_min"] for route in within_horizon),
                max(route["travel"]["total_duration_min"] for route in within_horizon),
            ] if within_horizon else []
        ),
        "requested_menu_items": menu_preferences,
        "requested_stalls": stall_preferences,
        "requested_cuisines": cuisine_preferences,
        "budget_sgd": budget,
    }

    location["selected_centres"] = [
        {
            "centre_id": route["centre_id"],
            "rank": rank,
            "logistics_fit": 1.0 / rank,
            "rationale": (
                "Nearest route-checked centre with a catalogue menu match."
                if selected_matches else
                "Nearest route-checked centre inside the search horizon; no matching menu item was found."
            ),
        }
        for rank, (route, _) in enumerate(selected_to_use, start=1)
    ]
    if outcome in {"matched_in_initial_shortlist", "expanded_match_found"}:
        search_info.update({
            "additional_centres": [
                centre_summary(route["centre_id"])
                for route, _ in selected_matches
            ],
            "matched_menu_item_count": sum(len(items) for _, items in selected_matches),
        })
        if outcome == "expanded_match_found":
            location["reasoning"] = (
                "The closest initial centres had no source-backed menu match, so the "
                "search expanded to the nearest route-checked centres that do."
            )
        location.setdefault("limitations", []).append(
            "Menu coverage is partial; expanded results are limited to route-checked centres with linked catalogue entries."
        )
    elif outcome == "farther_match_requires_approval":
        search_info.update({
            "farther_matches": [
                centre_summary(route["centre_id"])
                for route, _ in matched_farther
            ],
            "farther_match_item_count": sum(len(items) for _, items in matched_farther),
            "catalogue_is_partial": True,
        })
        location["reasoning"] = (
            "No matching menu item was found within the normal nearby-search horizon, "
            "but farther route-checked centres have catalogue matches."
        )
        location.setdefault("limitations", []).append(
            "Farther catalogue matches are shown for confirmation before expanding the trip."
        )
    else:
        search_info.update({"farther_matches": [], "catalogue_is_partial": True})
        location["reasoning"] = (
            "No source-backed menu match was found among the route-checked centres."
        )
        location.setdefault("limitations", []).append(
            "The stall catalogue is partial; absence from these results does not establish that the dish is unavailable elsewhere."
        )
    location["search_expansion"] = search_info
    return {"location_result": location}


def dispatch_workers(state: SharedState):
    if state.get("needs_location"):
        return "location_required_response"
    if (
        state.get("parsed_request", {}).get("intent") == "food_discovery"
        and not (state.get("location_result") or {}).get("selected_centres")
    ):
        return "verify_results"
    tasks = [task for task in state.get("tasks", []) if task.agent != "location"]
    if not tasks:
        return "verify_results"
    return [
        Send("worker", {
            "task": task,
            "user_request": state["user_request"],
            "conversation_context": state.get("conversation_context", {}),
            "parsed_request": state["parsed_request"],
            "user_location": state.get("user_location"),
            "user_location_source": state.get("user_location_source"),
            "location_result": state.get("location_result"),
        })
        for task in tasks
    ]


def worker(state: WorkerState) -> dict[str, Any]:
    task = state["task"]
    parsed = state.get("parsed_request", {})
    location = state.get("location_result") or {}
    centres = [
        item.get("centre_id")
        for item in location.get("selected_centres", [])
        if isinstance(item, dict) and item.get("centre_id")
    ]
    from services.location_service import load_hawker_centres
    centre_names_by_id = {
        item["id"]: item["name"] for item in load_hawker_centres()
    }
    centre_names = [
        centre_names_by_id[centre_id]
        for centre_id in centres
        if centre_id in centre_names_by_id
    ]
    if task.agent == "dietary":
        from agents.dietary_agent import run
        result = run(
            task.task,
            user_request=state.get("user_request"),
            dietary_preferences=parsed.get("dietary_preferences", []),
            cuisine_preferences=parsed.get("cuisine_preferences", []),
            menu_item_preferences=parsed.get("menu_item_preferences", []),
            stall_name_preferences=parsed.get("stall_name_preferences", []),
            excluded_menu_item_ids=parsed.get("excluded_menu_item_ids", []),
            candidate_centres=centre_names,
            candidate_centre_ids=centres,
        )
    elif task.agent == "budget":
        from agents.budget_agent import run
        result = run(
            task.task,
            budget=parsed.get("budget_amount"),
            excluded_menu_item_ids=parsed.get("excluded_menu_item_ids", []),
            candidate_centres=centre_names,
            candidate_centre_ids=centres,
        )
    elif task.agent == "queue":
        from agents.queue_agent import run
        result = run(
            task.task,
            user_request=state.get("user_request"),
            parsed_request=parsed,
            candidate_centres=centre_names,
            candidate_centre_ids=centres,
        )
    elif task.agent == "weather":
        from agents.weather_agent import run
        result = run(task.task, parsed_request=parsed, candidate_centres=centre_names)
    else:
        result = {"agent": task.agent, "error": "Unknown worker."}
    result.setdefault("task", task.task)
    result["candidate_centre_ids"] = centres
    result["status"] = "ok" if not result.get("error") else "error"
    return {"agent_results": [_result_envelope(task.agent, result)]}


def location_required_response(state: SharedState) -> dict[str, str]:
    error = (state.get("location_result") or {}).get("error")
    return {"final_recommendation": error or "Please share a starting location."}


def respond_without_workers(state: SharedState) -> dict[str, str]:
    intent = state.get("parsed_request", {}).get("intent")
    if intent == "smalltalk":
        reply = (
            "Hi! I can help find hawker options using your location, budget, "
            "dietary needs, and queue preferences."
        )
    else:
        reply = "I can help with hawker food recommendations. Tell me what you want and where to look."
    return {"final_recommendation": reply}


def clarification_response(state: SharedState) -> dict[str, str]:
    question = state.get("parsed_request", {}).get("clarification_question")
    return {"final_recommendation": question or "Could you clarify what you mean?"}


def _queue_status_response(user_request: str, verified: list[dict[str, Any]]) -> Optional[str]:
    if not _requests_queue_data(user_request):
        return None
    queue_result = next((
        item for item in verified
        if item.get("agent") == "queue" and item.get("verified")
    ), None)
    if not queue_result:
        return None

    queue_candidates = [
        item for item in queue_result.get("candidates", [])
        if isinstance(item, dict) and item.get("queue_minutes") is not None
    ]
    if not queue_candidates:
        return (
            "I couldn't find queue estimates for the route-shortlisted centres. "
            "The queue feed is simulated demo data, not a live service."
        )

    request_terms = _stall_name_terms(user_request)
    named_stalls = [
        item for item in queue_candidates
        if item.get("stall_name")
        and len(_stall_name_terms(item["stall_name"])) >= 2
        and _stall_name_terms(item["stall_name"]).issubset(request_terms)
    ]

    location = next((
        item.get("payload", {}) for item in verified
        if item.get("agent") == "location"
    ), {})
    selected = location.get("selected_centres", [])
    selected_ids = [
        item.get("centre_id") for item in selected
        if isinstance(item, dict) and item.get("centre_id")
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in queue_candidates:
        centre_id = candidate.get("hawker_centre_id")
        if selected_ids and centre_id not in selected_ids:
            continue
        grouped.setdefault(str(centre_id or candidate.get("hawker_centre") or ""), []).append(candidate)

    if named_stalls:
        selected_names = {item.get("stall_name") for item in named_stalls}
        direct_candidates = [
            candidate for rows in grouped.values() for candidate in rows
            if candidate.get("stall_name") in selected_names
        ]
        if direct_candidates:
            summary_lines = [
                "SIMULATED QUEUE DATA: These estimates are for demonstration only, not live conditions.",
                "",
                "Queue estimates for the stall you asked about:",
            ]
            for index, candidate in enumerate(direct_candidates, start=1):
                summary_lines.extend([
                    f"{index}. {candidate.get('hawker_centre', 'Hawker centre')}",
                    f"   - {candidate['stall_name']}: about {int(candidate['queue_minutes'])} minutes estimated wait.",
                    f"   - Crowd level: {candidate.get('crowd_level') or 'Unspecified'}.",
                ])
            return "\n".join(summary_lines)

    if selected_ids:
        ordered_groups = [
            (centre_id, grouped[centre_id])
            for centre_id in selected_ids if centre_id in grouped
        ]
    else:
        ordered_groups = list(grouped.items())
    if not ordered_groups:
        return (
            "I couldn't find queue estimates for the route-shortlisted centres. "
            "The queue feed is simulated demo data, not a live service."
        )

    centre_names = {}
    for rows in queue_candidates:
        if rows.get("hawker_centre_id"):
            centre_names[str(rows["hawker_centre_id"])] = rows.get("hawker_centre", "")

    origin = location.get("origin_label")
    heading = (
        f"Queue and crowd estimates near {origin}:"
        if origin else "Queue and crowd estimates for the nearby centres:"
    )
    summary_lines = [
        "SIMULATED QUEUE DATA: These estimates are for demonstration only, not live conditions.",
        "",
        heading,
    ]
    for index, (centre_id, rows) in enumerate(ordered_groups[:3], start=1):
        waits = [int(item["queue_minutes"]) for item in rows]
        crowd_counts: dict[str, int] = {}
        for item in rows:
            crowd = str(item.get("crowd_level") or "Unspecified")
            crowd_counts[crowd] = crowd_counts.get(crowd, 0) + 1
        most_common_crowd = max(
            crowd_counts,
            key=lambda crowd: (crowd_counts[crowd], crowd),
        )
        name = centre_names.get(centre_id) or rows[0].get("hawker_centre") or centre_id
        average_wait = round(sum(waits) / len(waits))
        summary_lines.extend([
            f"{index}. {name}",
            f"   - Average estimated wait: about {average_wait} minutes across {len(rows)} stalls.",
            f"   - Most common crowd level: {most_common_crowd}.",
        ])
    return "\n".join(summary_lines)


def verify_results_node(state: SharedState) -> dict[str, Any]:
    from agents.verifier import verify_results

    agent_results = list(state.get("agent_results", []))
    location_result = state.get("location_result")
    if location_result:
        replacement = _result_envelope("location", location_result)
        agent_results = [
            replacement if item.get("agent") == "location" else item
            for item in agent_results
        ]
    verified = verify_results(
        agent_results, state.get("parsed_request", {})
    )
    return {"verified_results": verified}


def synthesizer(state: SharedState) -> dict[str, str]:
    parsed = state.get("parsed_request", {})
    verified = state.get("verified_results", [])
    location = next((
        item.get("payload", {}) for item in verified
        if item.get("agent") == "location"
    ), {})
    interpretation = location.get("intent_interpretation", {})
    if (
        parsed.get("target_centre_text")
        and interpretation.get("target_resolution") == "not_found"
    ):
        target = parsed["target_centre_text"]
        prior_origin = (
            (state.get("conversation_context", {}).get("state", {}) or {})
            .get("origin_label")
        )
        nearby_offer = (
            f" If you want, I can search for nearby centres around {prior_origin}."
            if prior_origin else ""
        )
        return {"final_recommendation": (
            f"I couldn't verify {target} in the hawker-centre catalogue, so I "
            "can't confirm food options there. I haven't substituted other "
            f"centres for it.{nearby_offer} You can also name another centre."
        )}
    queue_reply = _queue_status_response(
        state.get("user_request", ""), verified
    )
    if queue_reply:
        return {"final_recommendation": queue_reply}
    if parsed.get("intent") == "food_discovery":
        joined = next((
            item for item in verified
            if item.get("agent") == "candidate_join"
        ), None)
        if not joined or not joined.get("candidates"):
            location = next((
                item.get("payload", {}) for item in verified
                if item.get("agent") == "location"
            ), {})
            has_food_filter = any((
                parsed.get("menu_item_preferences"),
                parsed.get("stall_name_preferences"),
                parsed.get("cuisine_preferences"),
                parsed.get("dietary_preferences"),
                parsed.get("budget_amount") is not None,
            ))
            if not has_food_filter:
                selected = location.get("selected_centres", [])
                routes = {
                    item.get("centre_id"): item
                    for item in location.get("logistics", [])
                    if isinstance(item, dict)
                }
                from services.location_service import load_hawker_centres
                names = {
                    item["id"]: item["name"]
                    for item in load_hawker_centres()
                }
                nearby = []
                for centre in selected[:3]:
                    if not isinstance(centre, dict):
                        continue
                    centre_id = centre.get("centre_id")
                    name = names.get(centre_id)
                    if not name:
                        continue
                    travel = (routes.get(centre_id) or {}).get("travel", {})
                    minutes = travel.get("total_duration_min")
                    mode = travel.get("mode")
                    if minutes is not None:
                        mode_label = "public transport" if mode == "pt" else mode
                        nearby.append(f"{name} (about {minutes} min by {mode_label})")
                    else:
                        nearby.append(name)
                if nearby:
                    origin = location.get("origin_label") or "your starting point"
                    return {"final_recommendation": (
                        f"The closest route-checked hawker centres near {origin} are: "
                        f"{'; '.join(nearby)}. I don't have menu listings for these "
                        "centres in the current stall catalogue, so I can't reliably "
                        "name their food options yet. That is a data coverage gap, "
                        "not evidence that they have no food stalls."
                    )}
            expansion = location.get("search_expansion", {})
            outcome = expansion.get("outcome")
            preference = (
                parsed.get("menu_item_preferences", [])
                or parsed.get("stall_name_preferences", [])
                or parsed.get("cuisine_preferences", [])
            )
            request_label = ", ".join(preference) if preference else "options matching your request"
            origin = location.get("origin_label") or parsed.get("origin_text") or "your starting point"
            horizon = expansion.get("search_horizon_min")

            if outcome == "farther_match_requires_approval":
                farther = expansion.get("farther_matches", [])
                nearest = farther[0] if farther else {}
                centre_name = nearest.get("centre_name") or "a farther hawker centre"
                minutes = nearest.get("travel_time_min")
                mode = nearest.get("travel_mode")
                mode_label = "public transport" if mode == "pt" else mode or "the selected travel mode"
                distance = f"about {minutes} minutes by {mode_label}" if minutes is not None else "farther away"
                reply = (
                    f"I couldn't find {request_label} within {horizon} minutes of {origin}. "
                    f"The nearest catalogue match I found is at {centre_name}, {distance}. "
                    f"Would you like me to show options there, look for other food within "
                    f"{horizon} minutes, or check a centre you name?"
                )
                return {"final_recommendation": reply}

            if outcome == "no_match_in_routed_pool":
                count = expansion.get("route_checked_centre_count", 0)
                if horizon is not None:
                    scope = f"within {horizon} minutes of {origin}"
                else:
                    scope = f"near {origin}"
                reply = (
                    f"I couldn't find {request_label} in the menu data for the "
                    f"{count} route-checked centres {scope}. The stall catalogue is "
                    f"partial, so this doesn't prove the dish isn't available nearby. "
                    f"Would you like other food options in that area, a farther search, "
                    f"or a search at a centre you name?"
                )
                return {"final_recommendation": reply}

            if outcome == "dietary_verification_unavailable":
                return {
                    "final_recommendation": (
                        "I can't verify that dietary restriction from this menu data: "
                        "it has no ingredient, allergen, or certification details. "
                        "Would you like to see menu options with that limitation made "
                        "clear, or search using another preference?"
                    )
                }

    from agents.recommendation_agent import run

    result = run(
        state["user_request"],
        verified,
        parsed_request=parsed,
    )
    joined = next((
        item for item in verified
        if item.get("agent") == "candidate_join"
    ), {})
    if any(
        candidate.get("is_mock")
        for candidate in joined.get("candidates", [])
        if isinstance(candidate, dict)
    ):
        result = (
            "DEMO DATA: Listings marked Demo and their prices are fictional "
            "examples, not verified stalls or current menu prices.\n\n"
            + result
        )
    return {"final_recommendation": result}


def build_workflow():
    graph = StateGraph(SharedState)
    graph.add_node("build_model", build_model)
    graph.add_node("parse_request", parse_request)
    graph.add_node("location_worker", location_worker)
    graph.add_node("reuse_location", reuse_location)
    graph.add_node("expand_menu_search", expand_menu_search)
    graph.add_node("worker", worker)
    graph.add_node("verify_results", verify_results_node)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("respond_without_workers", respond_without_workers)
    graph.add_node("clarification_response", clarification_response)
    graph.add_node("location_required_response", location_required_response)

    graph.add_edge(START, "build_model")
    graph.add_edge("build_model", "parse_request")
    graph.add_conditional_edges("parse_request", route_after_parse, {
        "respond_without_workers": "respond_without_workers",
        "clarification_response": "clarification_response",
        "location_worker": "location_worker",
        "reuse_location": "reuse_location",
    })
    graph.add_edge("location_worker", "expand_menu_search")
    graph.add_edge("reuse_location", "expand_menu_search")
    graph.add_conditional_edges("expand_menu_search", dispatch_workers, [
        "worker", "verify_results", "location_required_response",
    ])
    graph.add_edge("worker", "verify_results")
    graph.add_edge("verify_results", "synthesizer")
    graph.add_edge("synthesizer", END)
    graph.add_edge("respond_without_workers", END)
    graph.add_edge("clarification_response", END)
    graph.add_edge("location_required_response", END)
    return graph.compile()
