"""Programmatic checks applied to worker evidence before recommendation."""

from typing import Any

from services.location_service import load_hawker_centres
from services.hawker_service import get_menu_item_record, get_menu_stall_record


def _verified_location(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload", result)
    valid_ids = {item["id"] for item in load_hawker_centres()}
    all_logistics = [
        item for item in payload.get("logistics", [])
        if isinstance(item, dict) and item.get("centre_id") in valid_ids
    ]
    route_by_id = {item["centre_id"]: item for item in all_logistics}
    selected = [
        item for item in payload.get("selected_centres", [])
        if isinstance(item, dict)
        if item.get("centre_id") in valid_ids
    ]
    selected_ids = {item["centre_id"] for item in selected}
    logistics = [
        item for item in payload.get("logistics", [])
        if isinstance(item, dict)
        if item.get("centre_id") in valid_ids
        and item.get("centre_id") in selected_ids
    ]
    checked = {**payload, "selected_centres": selected, "logistics": logistics}
    expansion = payload.get("search_expansion")
    if isinstance(expansion, dict):
        expansion = dict(expansion)
        for key in ("initial_centres", "additional_centres", "farther_matches"):
            summaries = expansion.get(key)
            if not isinstance(summaries, list):
                continue
            valid_summaries = []
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                centre_id = summary.get("centre_id")
                route = route_by_id.get(centre_id)
                if not route:
                    continue
                measured_minutes = (route.get("travel") or {}).get("total_duration_min")
                if summary.get("travel_time_min") != measured_minutes:
                    continue
                valid_summaries.append(summary)
            expansion[key] = valid_summaries
        checked["search_expansion"] = expansion
    removed = len(payload.get("selected_centres", [])) - len(selected)
    checked["verified"] = removed == 0
    checked["verification"] = "centre IDs checked against the NEA catalogue"
    if removed:
        checked.setdefault("limitations", []).append(
            f"Removed {removed} centre(s) outside the NEA catalogue."
        )
    return {
        **result,
        "payload": checked,
        "candidates": [],
        "verified": checked["verified"],
        "verification": checked["verification"],
    }


def _verify_worker_result(
    result: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    payload = result.get("payload", result)
    agent = result.get("agent")
    candidates = [
        item for item in result.get("candidates", payload.get("candidates", []))
        if isinstance(item, dict)
    ]
    if agent in {"dietary", "budget"}:
        source_candidates = []
        for candidate in candidates:
            menu_item_id = candidate.get("menu_item_id")
            source = get_menu_item_record(str(menu_item_id or ""))
            if not source:
                continue
            if (
                candidate.get("stall_id") != source.get("stall_id")
                or candidate.get("hawker_centre_id") != source.get("hawker_centre_id")
            ):
                continue
            # Keep source-owned identity, name, menu and price fields. Workers
            # may add only their own matching explanation fields.
            worker_fields = (
                "matched_dietary_tags", "matched_cuisine", "preference_score",
                "reason", "budget_limit_sgd", "budget_match",
            )
            source_candidates.append({
                **source,
                **{key: candidate[key] for key in worker_fields if key in candidate},
            })
        candidates = source_candidates
    if agent == "dietary":
        diets = request.get("dietary_preferences", [])
        cuisines = [value.casefold() for value in request.get("cuisine_preferences", [])]
        if diets:
            candidates = [item for item in candidates if item.get("dietary_suitable") is True]
        if cuisines:
            candidates = [
                item for item in candidates
                if any(
                    any(
                        preference == category.casefold()
                        or preference in category.casefold().split()
                        for category in item.get("categories", [])
                    )
                    for preference in cuisines
                )
            ]
    elif agent == "budget" and request.get("budget_amount") is not None:
        limit = float(request["budget_amount"])
        candidates = [
            item for item in candidates
            if item.get("price_sgd_min") is not None
            and float(item["price_sgd_min"]) <= limit
        ]
    elif agent == "queue" and request.get("max_queue_min") is not None:
        limit = int(request["max_queue_min"])
        candidates = [
            item for item in candidates
            if item.get("stall_id")
            and item.get("queue_minutes") is not None
            and int(item["queue_minutes"]) <= limit
        ]
    if agent == "queue":
        candidates = [
            item for item in candidates
            if item.get("stall_id") and get_menu_stall_record(item["stall_id"])
        ]
    elif agent == "weather" and not payload.get("weather"):
        return {
            **result,
            "payload": payload,
            "verified": False,
            "status": "unavailable",
            "limitations": ["No weather evidence was returned."],
        }

    valid_candidates = [
        item for item in candidates
        if isinstance(item, dict) and item.get("stall_name")
    ]
    removed = len(candidates) - len(valid_candidates)
    checked = {
        **payload,
        "verified": True,
        "removed_candidate_count": removed,
        "verification": "required fields and parsed hard constraints checked",
    }
    return {
        **result,
        "candidates": valid_candidates,
        "verified": True,
        "removed_candidate_count": removed,
        "verification": checked["verification"],
        "payload": checked,
    }


def verify_results(
    agent_results: list[dict[str, Any]], parsed_request: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check catalogue identity and hard user constraints before synthesis."""
    verified = []
    for result in agent_results:
        if not isinstance(result, dict):
            continue
        if result.get("agent") == "location":
            verified.append(_verified_location(result))
        elif result.get("agent") in {"dietary", "budget", "queue", "weather"}:
            verified.append(_verify_worker_result(result, parsed_request))
        else:
            verified.append({
                **result,
                "verified": False,
                "status": "unavailable",
                "verification": "unrecognized worker result",
            })
    stall_results = [
        item for item in verified
        if item.get("agent") in {"dietary", "budget"}
        and item.get("verified")
    ]
    if stall_results:
        candidate_maps = []
        for result in stall_results:
            candidates = {}
            for candidate in result.get("candidates", []):
                item_id = candidate.get("menu_item_id")
                if item_id:
                    candidates[item_id] = candidate
            candidate_maps.append((result.get("agent"), candidates))

        common_keys = set(candidate_maps[0][1]) if candidate_maps else set()
        for _, candidates in candidate_maps[1:]:
            common_keys.intersection_update(candidates)
        # Preserve the catalogue ordering rather than relying on set order.
        ordered_keys = [
            item.get("menu_item_id") for item in stall_results[0].get("candidates", [])
            if item.get("menu_item_id") in common_keys
        ]
        queue_result = next(
            (item for item in verified if item.get("agent") == "queue" and item.get("verified")),
            None,
        )
        queue_by_stall = {
            item.get("stall_id"): item
            for item in (queue_result or {}).get("candidates", [])
            if item.get("stall_id")
        }
        max_queue = parsed_request.get("max_queue_min")
        joined = []
        for key in ordered_keys:
            combined = {}
            for agent, candidates in candidate_maps:
                combined.update(candidates[key])
            queue = queue_by_stall.get(combined.get("stall_id"))
            if max_queue is not None and (
                not queue
                or queue.get("queue_minutes") is None
                or int(queue["queue_minutes"]) > int(max_queue)
            ):
                continue
            if queue:
                combined.update({
                    "queue_minutes": queue.get("queue_minutes"),
                    "crowd_level": queue.get("crowd_level"),
                    "queue_score": queue.get("queue_score"),
                    "queue_source": queue.get("source"),
                })
            combined["supporting_agents"] = [agent for agent, _ in candidate_maps]
            joined.append(combined)
        joined = joined[:30]
        joined_result = {
            "agent": "candidate_join",
            "verified": True,
            "candidates": joined,
            "limitations": (
                [] if joined else [
                    "No menu items remain after applying the verified catalogue and worker constraints."
                ]
            ),
            "verification": "menu item IDs and centre IDs checked against the source-backed catalogue",
        }
        verified.append(joined_result)
    return verified
