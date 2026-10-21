"""Programmatic checks applied to worker evidence before recommendation."""

from typing import Any

from services.location_service import load_hawker_centres
from services.hawker_service import (
    VERIFIABLE_DIETS, _normalise, get_menu_item_record, get_menu_stall_record,
)


def _verified_location(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload", result)
    centres = {item["id"]: item for item in load_hawker_centres()}
    valid_ids = set(centres)
    all_logistics = []
    for item in payload.get("logistics", []):
        if not isinstance(item, dict) or item.get("centre_id") not in valid_ids:
            continue
        source = centres[item["centre_id"]]
        coords = item.get("coordinates") or {}
        travel = item.get("travel") or {}
        if not isinstance(travel, dict) or not isinstance(coords, dict):
            continue
        try:
            coordinates_match = (
                abs(float(coords["lat"]) - float(source["latitude"])) < 0.00001
                and abs(float(coords["lng"]) - float(source["longitude"])) < 0.00001
            )
        except (KeyError, TypeError, ValueError):
            coordinates_match = False
        if (
            item.get("centre_name") != source["name"]
            or item.get("address") != source["address"]
            or not coordinates_match
            or travel.get("source") not in {"onemap", "unavailable"}
            or travel.get("mode") not in {"pt", "walk", "drive", "cycle"}
        ):
            continue
        all_logistics.append(item)
    route_by_id = {item["centre_id"]: item for item in all_logistics}
    selected = []
    seen_ids = set()
    for item in payload.get("selected_centres", []):
        if not isinstance(item, dict) or item.get("centre_id") not in valid_ids:
            continue
        centre_id = item["centre_id"]
        if centre_id in seen_ids:
            continue
        seen_ids.add(centre_id)
        # Keep catalogue-owned centre identity and only a bounded rank. Drop
        # free-form model rationales so they cannot become response evidence.
        selected.append({
            "centre_id": centre_id,
            "rank": len(selected) + 1,
            "logistics_fit": 1.0,
            "rationale": (
                "Selected from verified catalogue and route evidence."
                if centre_id in route_by_id
                and (route_by_id[centre_id].get("travel") or {}).get("source") == "onemap"
                and (route_by_id[centre_id].get("travel") or {}).get("total_duration_min") is not None
                else "Selected from the verified centre catalogue; route evidence is unavailable."
            ),
        })
    selected_ids = {item["centre_id"] for item in selected}
    logistics = [
        item for item in all_logistics
        if item["centre_id"] in selected_ids
    ]
    checked = {**payload, "selected_centres": selected, "logistics": logistics}
    merchant_entries = payload.get("merchant_stalls")
    if isinstance(merchant_entries, list):
        # Re-derive the merchant list from the centre record, so a worker
        # cannot name a stall the directory does not list.
        rebuilt = []
        for entry in merchant_entries:
            if not isinstance(entry, dict):
                continue
            centre = centres.get(entry.get("centre_id"))
            if not centre:
                continue
            source_merchants = centre.get("merchant_stalls") or []
            wanted = {
                (item.get("name"), item.get("unit_number"))
                for item in entry.get("merchants", [])
                if isinstance(item, dict)
            }
            merchants = [
                {
                    "name": merchant.get("name"),
                    "unit_number": merchant.get("unit_number"),
                }
                for merchant in source_merchants
                if (merchant.get("name"), merchant.get("unit_number")) in wanted
            ]
            if not merchants:
                continue
            rebuilt.append({
                "centre_id": centre["id"],
                "centre_name": centre["name"],
                "merchants": merchants,
                "merchant_count": entry.get("merchant_count"),
                "source": centre.get("merchant_source"),
                "as_of": centre.get("merchant_as_of"),
                "note": entry.get("note"),
            })
        checked["merchant_stalls"] = rebuilt
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
            or candidate.get("queue_is_simulated") is not True
            or candidate.get("source") != "simulated_queue_estimate"
            or not isinstance(candidate.get("queue_estimate_notice"), str)
            or "not a live" not in candidate["queue_estimate_notice"].casefold()
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
                    if isinstance(preference, str) and any(
                        preference.casefold() == str(category).casefold()
                        or preference.casefold() in str(category).casefold().split()
                        for category in categories
                    )
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
    if agent == "dietary":
        diets = request.get("dietary_preferences", [])
        single_diet = request.get("dietary_preference")
        if single_diet:
            diets = list(diets) + [single_diet]
        cuisines = [value.casefold() for value in request.get("cuisine_preferences", [])]
        # Verifiable restrictions (vegetarian, vegan, gluten-free, etc.) require
        # structured data the source lacks, so they short-circuit to empty.
        # Inferred heuristics (halal, healthy, unhealthy) are not matched
        # against catalogue entries — they are not certification data.
        verifiable_diets = [d for d in diets if d in VERIFIABLE_DIETS]
        if verifiable_diets:
            candidates = []
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
    if agent == "queue":
        submitted_count = len(candidates)
        candidates, queue_limitations, valid_request = _verify_queue_candidates(
            candidates, request
        )
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
            "and parsed hard limit checked"
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
    return {
        **result,
        "candidates": valid_candidates,
        "verified": True,
        "removed_candidate_count": removed,
        "verification": checked["verification"],
        "confidence": result.get("confidence", 0.5),
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
