"""Programmatic checks applied to worker evidence before recommendation."""

import math
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from agents.location_grounding import (
    _LocationGrounding, _RankingOutputError, _matching_snapshot, _reconcile_grounding,
    _render_verified_choice, _usable_primary_routes,
)
from models.location_models import LocationDecision, TravelEstimate

from services.location_service import load_hawker_centres
from services.hawker_service import (
    VERIFIABLE_DIETS, INFERRED_DIETS, _matches_category, get_menu_item_record, get_menu_stall_record,
    find_dietary_stalls, find_stalls_within_budget,
)


def _nonnegative_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value >= 0 and (isinstance(value, int) or math.isfinite(value)))


def _eligible_location_routes(payload, centres, maximum):
    # Detect conflicting copies before discarding bad identities or measurements.
    usable, _ = _usable_primary_routes(payload.get("logistics") or [])
    eligible = {}
    for centre_id, item in usable.items():
        source = centres.get(centre_id)
        coords, travel = item.get("coordinates"), item["travel"]
        if not source or not isinstance(coords, dict):
            continue
        if item.get("centre_name") != source["name"] or item.get("address") != source["address"]:
            continue
        if any(
            not _nonnegative_number(coords.get(axis))
            or abs(coords[axis] - float(source[field])) >= 0.00001
            for axis, field in (("lat", "latitude"), ("lng", "longitude"))
        ):
            continue
        if any(travel.get(metric) is not None and not _nonnegative_number(travel[metric])
               for metric in ("walking_distance_m", "walking_time_min", "transfers")):
            continue
        transfers = travel.get("transfers")
        if transfers is not None and transfers != int(transfers):
            continue
        if any(travel.get(metric) is not None and travel[metric] != int(travel[metric])
               for metric in ("total_duration_min", "walking_time_min")):
            continue
        if maximum is not None and travel["total_duration_min"] > maximum:
            continue
        eligible[centre_id] = item
    return eligible


def _public_route(item, source):
    # Keep source-owned identity and verified measurements. Free-form provider
    # summaries/displays are not evidence for shelter, accessibility or walking.
    travel = {key: item["travel"].get(key) for key in (
        "total_duration_min", "walking_distance_m", "walking_time_min", "transfers", "mode", "source",
    )}
    estimate = TravelEstimate(
        **{key: value for key, value in travel.items() if key != "total_duration_min"}, confidence=0.5,
    )
    travel.update({"confidence": 0.5, "walking_time_min": estimate.walking_time_min,
                   "walking_distance_display": estimate.walking_distance_display,
                   "walking_time_display": estimate.walking_time_display,
                   "limitation": "Route confidence is a verifier policy value, not calibrated accuracy.",
                   "itinerary_summary": None})
    if estimate.walking_time_min is not None:
        travel["walking_time_display"] = (f"about {estimate.walking_time_min} min"
                                          if estimate.walking_time_min else "under 1 min")
    return {"centre_id": source["id"], "centre_name": source["name"], "address": source["address"],
            "coordinates": {"lat": float(source["latitude"]), "lng": float(source["longitude"])},
            "travel": travel, "alternative_travel": None,
            "evidence": ["Catalogue identity and primary route measurements checked by the verifier."],
            "limitations": (["Walking time is an approximate estimate, not measured walking duration."]
                            if travel["walking_time_min"] is not None else [])}


def _verified_location(result: dict[str, Any], parsed_request=None, *, grounding=None,
                       request_id=None, location_reused=False) -> dict[str, Any]:
    payload = result.get("payload", result)
    request = parsed_request or {}
    centres = {item["id"]: item for item in load_hawker_centres()}
    maximum = request.get("max_travel_time_min")
    valid_maximum = maximum is None or (isinstance(maximum, int) and not isinstance(maximum, bool) and maximum >= 0)
    bundle = None
    if not location_reused and isinstance(grounding, _LocationGrounding) and grounding.request_id == request_id:
        try:
            # Validate structure without coercing/replacing the server's captured facts.
            _LocationGrounding.model_validate(grounding.model_dump())
        except ValidationError:
            pass
        else:
            bundle = _reconcile_grounding(grounding, payload, request_id)
            if bundle.max_travel_time_min is not None:
                maximum = min(maximum, bundle.max_travel_time_min) if valid_maximum and maximum is not None else bundle.max_travel_time_min
    named_lookup = (request.get("target_centre_text") and request.get("intent") != "directions"
                    and not payload.get("origin") and not payload.get("logistics"))
    catalogue_only = bool(location_reused or parsed_request is None or named_lookup
                          or request.get("intent") == "centre_details"
                          or (isinstance(request.get("search_scope_mode"), str)
                              and request["search_scope_mode"] in {"global", "region"}))
    expansion = payload.get("search_expansion")
    horizon = expansion.get("search_horizon_min") if isinstance(expansion, dict) else None
    valid_horizon = horizon is None or (isinstance(horizon, int) and not isinstance(horizon, bool) and horizon >= 0)
    routes = (_eligible_location_routes(payload, centres, maximum)
              if valid_maximum and not catalogue_only else {})
    matched = ({route.centre_id: route for route in bundle.routes
                if route.centre_id in routes and _matching_snapshot(route, routes[route.centre_id])}
               if bundle is not None else {})
    names = {centre_id: source["name"] for centre_id, source in centres.items()}
    originals = {item.centre_id: item for item in bundle.selections} if bundle else {}
    supplements = {item.selection.centre_id: item for item in bundle.supplements} if bundle else {}
    proposed = payload.get("selected_centres") or []
    proposed_originals = [item.get("centre_id") for item in proposed if isinstance(item, dict)
                          and isinstance(item.get("centre_id"), str) and item["centre_id"] in originals]
    original_order = [item.centre_id for item in bundle.selections
                      if item.centre_id in proposed_originals] if bundle else []
    model_order_matches = proposed_originals == original_order
    selected = []
    seen_ids = set()
    sources = []
    for item in proposed:
        if (not isinstance(item, dict) or not isinstance(item.get("centre_id"), str)
                or item["centre_id"] not in centres):
            continue
        centre_id = item["centre_id"]
        if centre_id in seen_ids:
            continue
        seen_ids.add(centre_id)
        if not valid_maximum or (centre_id not in routes and (not catalogue_only or maximum is not None)):
            continue
        if not valid_horizon or (centre_id in routes and horizon is not None
                                 and routes[centre_id]["travel"]["total_duration_min"] > horizon):
            continue
        if bundle is not None:
            original = originals.get(centre_id)
            if centre_id not in originals and centre_id not in supplements:
                continue  # A comparison-only route is not a new selection authorization.
            if original is not None and original.status_reason == "not_selected":
                continue  # Menu-removed selections cannot regain approval through public metadata.
        fit, source = 0.0, "catalogue_only"
        rationale = "Selected from the verified centre catalogue; no fresh route ranking was verified."
        if centre_id in routes:
            minutes = routes[centre_id]["travel"]["total_duration_min"]
            rationale = f"Verified primary route: {minutes} min. Ranking proof was missing or rejected; model fit was not retained."
            source = "unproven"
            original = originals.get(centre_id)
            if original is not None and model_order_matches:
                try:
                    rationale = _render_verified_choice(bundle, original, matched, names)
                except _RankingOutputError:
                    pass
                else:
                    fit, source = original.logistics_fit, original.source
            supplement = supplements.get(centre_id)
            if (source == "unproven" and supplement is not None and centre_id in bundle.active_order
                    and supplement.selection.source == "menu_expansion"
                    and supplement.selection.status == "active" and supplement.selection.original_rank is None
                    and supplement.selection.route_id == supplement.route.route_id
                    and supplement.selection.fit_basis == "not_location_ranked"
                    and supplement.selection.logistics_fit is None and not supplement.selection.reasons
                    and _matching_snapshot(supplement.route, routes[centre_id])):
                rationale = f"Deterministic menu-expansion selection; verified OneMap primary route: {minutes} min."
                fit, source = 0.5, "menu_expansion"
        selected.append({
            "centre_id": centre_id, "rank": len(selected) + 1,
            "logistics_fit": fit, "rationale": rationale,
        })
        sources.append(source)
        if not catalogue_only and len(selected) == 3:
            break
    whole_basis = bool(bundle and selected and not bundle.supplements
                       and [item["centre_id"] for item in selected] == [item.centre_id for item in bundle.selections]
                       and len(selected) == len(proposed) and len(matched) == len(bundle.routes)
                       and all(source == bundle.decision_source for source in sources))
    confidence = bundle.confidence if whole_basis else 0.5 if selected else 0.0
    basis = ("Model-reported confidence is not calibrated." if whole_basis and bundle.decision_source == "grounded_model"
             else "Confidence is a conservative policy value, not calibrated accuracy.")
    provenance = {"grounded_model": "Location model-ranked", "deterministic_fallback": "Location deterministic fallback",
                  "menu_expansion": "deterministic menu expansion", "unproven": "route facts only; ranking proof not accepted",
                  "catalogue_only": "catalogue only; no fresh travel proof"}
    summary = "; ".join(f"{names[item['centre_id']]}: {provenance[source]}"
                        for item, source in zip(selected[:8], sources[:8]))
    if whole_basis:
        reasoning = "Verified Location selection. " + summary
    elif selected:
        reasoning = "Verifier rebuilt the selection after filtering, expansion, or missing/rejected ranking proof. " + summary
    else:
        reasoning = "No eligible Location selection remains after verification."
    limitations = [basis]
    if "unproven" in sources:
        limitations.append("Rejected or missing model explanations and fit scores were replaced with route facts and policy fit 0.")
    if "menu_expansion" in sources:
        limitations.append("Menu supplements use policy fit 0.5 and were not ranked by the Location model.")
    if "catalogue_only" in sources:
        limitations.append("Catalogue-only selections do not establish current travel times or travel fit.")
    if location_reused:
        limitations.append("Reused shortlist: cached routes and prose are not fresh route proof.")
    if maximum is not None:
        limitations.append("A hard travel-time maximum requires eligible current route evidence; cached or missing routes cannot satisfy it.")
    checked = {key: deepcopy(value) for key, value in payload.items()
               if key in LocationDecision.model_fields or key in {"agent", "error", "location_required", "search_expansion"}}
    checked.update(selected_centres=selected,
                   logistics=[_public_route(routes[item["centre_id"]], centres[item["centre_id"]])
                              for item in selected if item["centre_id"] in routes],
                   reasoning=reasoning[:1600], confidence=confidence, limitations=limitations)
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
                route = routes.get(centre_id) if isinstance(centre_id, str) else None
                if not route:
                    continue
                measured_minutes = (route.get("travel") or {}).get("total_duration_min")
                if summary.get("travel_time_min") != measured_minutes:
                    continue
                valid_summaries.append({"centre_id": centre_id, "centre_name": names[centre_id],
                                        "travel_time_min": measured_minutes})
            expansion[key] = valid_summaries
        expansion.pop("initial_location_reasoning", None)
        if not isinstance(expansion.get("outcome"), str) or expansion["outcome"] not in {"matched_in_initial_shortlist", "expanded_match_found",
                                            "farther_match_requires_approval", "no_match_in_routed_pool",
                                            "dietary_verification_unavailable"}:
            expansion = {}
        checked["search_expansion"] = expansion
        checked["limitations"].append("Menu coverage is partial; verification does not establish exhaustive dish availability.")
    removed = len(proposed) - len(selected)
    checked["verified"] = bool(selected) and removed == 0
    checked["verification"] = "catalogue identity, primary-route eligibility and accepted grounding checked"
    if removed:
        checked.setdefault("limitations", []).append(
            f"Removed {removed} centre selection(s) that lacked eligible catalogue or route evidence."
        )
    return {
        **result,
        "payload": checked,
        "candidates": [],
        "confidence": confidence,
        "limitations": checked["limitations"],
        "status": (result["status"] if isinstance(result.get("status"), str) and result["status"] in {"error", "needs_input", "unavailable"}
                   else result.get("status", "ok") if checked["verified"] else "partial"),
        "verified": checked["verified"],
        "verification": checked["verification"],
    }


def _verify_queue_candidates(
    candidates: list[dict[str, Any]], request: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], bool]:
    raw_limit = request.get("max_queue_min")
    if raw_limit is not None and (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or raw_limit < 0
    ):
        return [], [
            "Queue candidates were rejected because max_queue_min must be a non-negative whole number."
        ], False

    verified = []
    rejected_count = 0
    crowd_for_wait = (
        (5, "Low"),
        (15, "Moderate"),
        (25, "High"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejected_count += 1
            continue
        stall_id = candidate.get("stall_id")
        source = get_menu_stall_record(str(stall_id or ""))
        wait = candidate.get("queue_minutes")
        queue_score = candidate.get("queue_score")
        crowd = candidate.get("crowd_level")
        if (
            not source
            or not source.get("hawker_centre_id")
            or candidate.get("hawker_centre_id") != source.get("hawker_centre_id")
            or candidate.get("queue_is_mock") is not True
            or candidate.get("source") != "synthetic_queue_mock"
            or not isinstance(candidate.get("queue_mock_notice"), str)
            or "not a live" not in candidate["queue_mock_notice"].casefold()
            or isinstance(wait, bool)
            or not isinstance(wait, int)
            or wait < 0
            or isinstance(queue_score, bool)
            or not isinstance(queue_score, int)
            or not 0 <= queue_score <= 100
        ):
            rejected_count += 1
            continue
        expected_crowd = next(
            (level for threshold, level in crowd_for_wait if wait <= threshold),
            "Very High",
        )
        if crowd != expected_crowd:
            rejected_count += 1
            continue
        if raw_limit is not None and wait > raw_limit:
            rejected_count += 1
            continue
        verified.append({
            **candidate,
            "stall_id": source["stall_id"],
            "stall_name": source["stall_name"],
            "hawker_centre_id": source["hawker_centre_id"],
            "hawker_centre": source["hawker_centre"],
        })

    limitations = []
    if rejected_count:
        limitations.append(
            f"Rejected {rejected_count} queue candidate(s) with invalid source identity, "
            "synthetic provenance, queue fields, or constraint values."
        )
    return verified, limitations, True


def _verify_worker_result(
    result: dict[str, Any], request: dict[str, Any], *, approved_centres=None, allowed_menu_ids=None,
) -> dict[str, Any]:
    payload = result.get("payload", result)
    agent = result.get("agent")
    candidates = [
        item for item in result.get("candidates", payload.get("candidates", []))
        if isinstance(item, dict)
    ]
    food_submitted_count = len(candidates)
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
                "dietary_suitable", "dietary_confidence",
                "reason", "budget_limit_sgd", "budget_match",
            )
            worker_values = {
                key: candidate[key] for key in worker_fields if key in candidate
            }
            tags = set(source.get("dietary_tags", []))
            if "matched_dietary_tags" in worker_values:
                requested_tags = worker_values["matched_dietary_tags"]
                if not isinstance(requested_tags, list):
                    requested_tags = []
                worker_values["matched_dietary_tags"] = [
                    tag for tag in requested_tags
                    if isinstance(tag, str) and tag in tags
                ]
            if "matched_cuisine" in worker_values:
                categories = source.get("categories", [])
                requested_cuisine = request.get("cuisine_preferences", [])
                worker_values["matched_cuisine"] = [
                    preference for preference in requested_cuisine
                    if isinstance(preference, str) and _matches_category(preference, categories)
                ]
            worker_values["reason"] = "Matched verified catalogue fields and request constraints."
            if agent == "budget":
                amount = request.get("budget_amount")
                minimum = source.get("price_sgd_min")
                maximum = source.get("price_sgd_max")
                worker_values["budget_limit_sgd"] = amount
                worker_values["budget_match"] = (
                    "priced_within_budget"
                    if amount is not None and maximum is not None and maximum <= amount
                    else "starting_price_within_budget"
                    if amount is not None and minimum is not None and minimum <= amount
                    else "not_within_budget"
                )
            source_candidates.append({
                **source,
                **worker_values,
            })
        candidates = source_candidates
        if approved_centres is not None:
            candidates = [item for item in candidates if item.get("hawker_centre_id") in approved_centres]
        if allowed_menu_ids is not None:
            candidates = [item for item in candidates if item.get("menu_item_id") in allowed_menu_ids]
    # The entry point's source matcher already applied all food constraints.
    # Retain legacy checks only for direct callers without a matching-ID set.
    if agent == "dietary" and allowed_menu_ids is None:
        diets = request.get("dietary_preferences", [])
        single_diet = request.get("dietary_preference")
        if single_diet:
            diets = list(diets) + [single_diet]
        cuisines = [value.casefold() for value in request.get("cuisine_preferences", [])]
        # Verifiable restrictions (vegetarian, vegan, gluten-free, etc.) require
        # structured data the source lacks, so they short-circuit to empty.
        # Inferred heuristics (halal, healthy, unhealthy) are matched against
        # each item's inferred dietary_tags.
        verifiable_diets = [d for d in diets if d in VERIFIABLE_DIETS and d not in INFERRED_DIETS]
        inferred_diets = [d for d in diets if d in INFERRED_DIETS]
        if verifiable_diets:
            candidates = []
        if inferred_diets:
            candidates = [
                item for item in candidates
                if any(d in item.get("dietary_tags", []) for d in inferred_diets)
            ]
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
    elif agent == "budget" and allowed_menu_ids is None and request.get("budget_amount") is not None:
        limit = float(request["budget_amount"])
        candidates = [
            item for item in candidates
            if item.get("price_sgd_min") is not None
            and float(item["price_sgd_min"]) <= limit
        ]
    if agent == "queue":
        submitted_count = len(candidates)
        candidates, queue_limitations, valid_request = _verify_queue_candidates(
            candidates, request
        )
        if approved_centres is not None:
            candidates = [item for item in candidates if item["hawker_centre_id"] in approved_centres]
        if not valid_request:
            return {
                **result,
                "candidates": [],
                "verified": False,
                "status": "unavailable",
                "confidence": 0.0,
                "limitations": list(dict.fromkeys(
                    result.get("limitations", []) + queue_limitations
                )),
                "verification": "queue request constraints failed validation",
                "payload": payload,
            }
        verification = (
            "stall and centre identity, simulated queue provenance, queue values, "
            "parsed hard limit, and applicable Location approval checked"
        )
        confidence = (
            round(len(candidates) / submitted_count, 2)
            if submitted_count else 0.0
        )
        payload = {
            **payload,
            "confidence_basis": (
                "Fraction of submitted queue records that passed source and constraint "
                "checks; this is evidence completeness, not prediction accuracy."
            ),
        }
        result = {
            **result,
            "limitations": list(dict.fromkeys(
                result.get("limitations", []) + queue_limitations
            )),
            "confidence": confidence,
        }
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
    removed = (
        submitted_count - len(valid_candidates)
        if agent == "queue"
        else food_submitted_count - len(valid_candidates) if agent in {"dietary", "budget"}
        else len(candidates) - len(valid_candidates)
    )
    checked = {
        **payload,
        "verified": True,
        "removed_candidate_count": removed,
        "verification": (
            verification if agent == "queue"
            else "required fields and parsed hard constraints checked"
        ),
    }
    if agent in {"dietary", "budget", "queue"} and "candidates" in checked:
        checked["candidates"] = valid_candidates
    return {
        **result,
        "candidates": valid_candidates,
        "verified": True,
        "removed_candidate_count": removed,
        "verification": checked["verification"],
        "confidence": result.get("confidence", 0.5),
        "payload": checked,
    }


def _matching_menu_ids(request, approved_centres):
    if approved_centres is not None and not approved_centres:
        return set()  # The source service interprets [] as unrestricted search.
    scope = sorted(approved_centres) if approved_centres is not None else None
    diets = list(request.get("dietary_preferences", []))
    if request.get("dietary_preference"):
        diets.append(request["dietary_preference"])
    matches = find_dietary_stalls(
        dietary_preferences=diets, cuisine_preferences=request.get("cuisine_preferences", []),
        menu_item_preferences=request.get("menu_item_preferences", []),
        stall_name_preferences=request.get("stall_name_preferences", []),
        excluded_menu_item_ids=request.get("excluded_menu_item_ids", []), candidate_centre_ids=scope,
    )
    ids = {item["menu_item_id"] for item in matches if item.get("menu_item_id")}
    if request.get("budget_amount") is not None:
        ids.intersection_update(item["menu_item_id"] for item in find_stalls_within_budget(
            budget=request["budget_amount"], candidate_centre_ids=scope,
            excluded_menu_item_ids=request.get("excluded_menu_item_ids", []),
        ) if item.get("menu_item_id"))
    return ids


def verify_results(
    agent_results: list[dict[str, Any]], parsed_request: dict[str, Any], *,
    location_grounding=None, request_id=None, location_reused=False,
) -> list[dict[str, Any]]:
    """Check catalogue identity and hard user constraints before synthesis."""
    # Verify Location first, independent of worker/envelope ordering.
    locations = {index: _verified_location(
        result, parsed_request, grounding=location_grounding, request_id=request_id,
        location_reused=location_reused,
    ) for index, result in enumerate(agent_results)
        if isinstance(result, dict) and result.get("agent") == "location"}
    has_food = any(isinstance(result, dict) and result.get("agent") in {"dietary", "budget"}
                   for result in agent_results)
    # Normalized queue-only requests may use food_discovery without a Location
    # envelope. Only genuinely unscoped queue lookups get the None sentinel.
    unscoped_queue = (not locations and not has_food and parsed_request.get("intent") in {"queue", "food_discovery"}
                      and any(isinstance(result, dict) and result.get("agent") == "queue" for result in agent_results)
                      and not any((parsed_request.get("origin_text"), parsed_request.get("requested_area"),
                                   parsed_request.get("max_travel_time_min") is not None,
                                   parsed_request.get("menu_item_preferences"), parsed_request.get("cuisine_preferences"),
                                   parsed_request.get("dietary_preferences"), parsed_request.get("dietary_preference"),
                                   parsed_request.get("excluded_menu_item_ids"), parsed_request.get("budget_amount") is not None)))
    scoped = bool(locations) or (not unscoped_queue and parsed_request.get("intent") in {"queue", "food_discovery", "centre_details", "directions"})
    approved_centres = None
    if scoped:
        scopes = [{item["centre_id"] for item in result["payload"]["selected_centres"]}
                  for result in locations.values()]
        approved_centres = set.intersection(*scopes) if scopes else set()
    allowed_menu_ids = _matching_menu_ids(parsed_request, approved_centres) if has_food else None
    verified = []
    for index, result in enumerate(agent_results):
        if not isinstance(result, dict):
            continue
        if result.get("agent") == "location":
            verified.append(locations[index])
        elif result.get("agent") in {"dietary", "budget", "queue", "weather"}:
            verified.append(_verify_worker_result(
                result, parsed_request, approved_centres=approved_centres, allowed_menu_ids=allowed_menu_ids,
            ))
        elif result.get("agent") == "candidate_join":
            # Only this verifier may create the derived join consumed by synthesis.
            continue
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
            if ((approved_centres is not None and combined.get("hawker_centre_id") not in approved_centres)
                    or combined.get("menu_item_id") not in allowed_menu_ids):
                continue
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
