"""Bounded Recommendation plans and deterministic rendering of accepted facts."""

import math
from typing import Annotated, Literal, Tuple

from pydantic import BaseModel, Field, StrictStr

from agents.recommendation_context import _RecommendationContext
from models.location_models import TravelEstimate
from services.hawker_service import get_menu_item_record
from services.location_service import load_hawker_centres


_NO_FOOD = "No verified eligible food options remain. The menu catalogue is partial; this does not prove matching food is unavailable elsewhere."
_Reason = Literal["eligible_menu", "listed_within_budget", "starting_price_within_budget", "verified_queue"]


class _MenuChoice(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    menu_item_id: Annotated[StrictStr, Field(min_length=1, max_length=160)]
    reason_refs: Tuple[_Reason, ...] = Field(default=("eligible_menu",), min_length=1, max_length=3)


class _RecommendationPlan(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    choices: Tuple[_MenuChoice, ...] = Field(min_length=1, max_length=30)


class _InvalidPlan(ValueError):
    pass


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _label(value):
    return " ".join(str(value or "").split())[:240]


def _eligible_rows(context, agent_results):
    """Intersect the immutable approval snapshot, current join and catalogue.

    Re-read only local catalogue data. A changed identity or listed price cannot
    silently inherit eligibility from an earlier join snapshot.
    """
    joined = next((item for item in agent_results if isinstance(item, dict)
                   and item.get("agent") == "candidate_join" and item.get("verified") is True), {})
    ownership = {item.menu_item_id: item for item in context.eligible_menu_items}
    centres = {item.centre_id for item in context.centres}
    protected = ("menu_item_id", "stall_id", "hawker_centre_id", "hawker_centre", "stall_name",
                 "menu_item_name", "price_sgd_min", "price_sgd_max", "price_label", "is_mock",
                 "categories", "dietary_tags", "unit_number", "unit", "price_status",
                 "price_sgd_options", "price_option_labels")
    rows, seen = [], set()
    for item in joined.get("candidates", [])[:30]:
        if not isinstance(item, dict) or not isinstance(item.get("menu_item_id"), str):
            continue
        key = item["menu_item_id"]
        proof = ownership.get(key)
        source = get_menu_item_record(key) if proof is not None else None
        if (source is None or key in seen
                or proof.centre_id not in centres
                or any(source.get(field) is not None and not _number(source[field])
                       for field in ("price_sgd_min", "price_sgd_max"))
                or source.get("hawker_centre_id") != proof.centre_id or source.get("stall_id") != proof.stall_id
                or any(item.get(field) != source.get(field) for field in protected)
                or (context.approved_centre_ids is not None and proof.centre_id not in context.approved_centre_ids)):
            continue
        seen.add(key)
        row = dict(source)
        # Queue fields here are from the verifier-owned join, never the model.
        if item.get("queue_source") == "synthetic_queue_mock" and _number(item.get("queue_minutes")):
            row.update(queue_minutes=item["queue_minutes"], queue_source=item["queue_source"])
        rows.append(row)
    return rows


def _reason_refs(row, request):
    refs = {"eligible_menu"}
    budget = request.get("budget_amount")
    if _number(budget):
        if _number(row.get("price_sgd_max")) and row["price_sgd_max"] <= budget:
            refs.add("listed_within_budget")
        elif _number(row.get("price_sgd_min")) and row["price_sgd_min"] <= budget:
            refs.add("starting_price_within_budget")
    if row.get("queue_source") == "synthetic_queue_mock" and _number(row.get("queue_minutes")):
        refs.add("verified_queue")
    return refs


def _centre_order(context, rows):
    represented = dict.fromkeys(item["hawker_centre_id"] for item in rows)
    if context.approved_centre_ids is None:
        return list(represented)
    return [centre for centre in context.approved_centre_ids if centre in represented]


def _validate_plan(plan, context, rows, request):
    by_id = {item["menu_item_id"]: item for item in rows}
    seen, counts = set(), {}
    for choice in plan.choices:
        row = by_id.get(choice.menu_item_id)
        if row is None or choice.menu_item_id in seen:
            raise _InvalidPlan("unknown_or_duplicate_menu_id")
        if (len(set(choice.reason_refs)) != len(choice.reason_refs)
                or not set(choice.reason_refs) <= _reason_refs(row, request)):
            raise _InvalidPlan("unsupported_reason_reference")
        seen.add(choice.menu_item_id)
        centre = row["hawker_centre_id"]
        counts[centre] = counts.get(centre, 0) + 1
        if counts[centre] > 3:
            raise _InvalidPlan("too_many_items_per_centre")
    if set(counts) != set(_centre_order(context, rows)):
        raise _InvalidPlan("missing_eligible_centre")
    return plan


def _fallback_plan(context, rows):
    return _RecommendationPlan(choices=tuple(_MenuChoice(menu_item_id=next(
        row["menu_item_id"] for row in rows if row["hawker_centre_id"] == centre))
        for centre in _centre_order(context, rows)))


def _route_lines(centre):
    route = centre.primary_route
    if route is None:
        return ["No fresh route evidence; catalogue identity does not establish proximity or travel eligibility."]
    mode = "public transport" if route.mode == "pt" else route.mode
    duration = route.total_duration_min.value
    lines = [f"Verified primary route: {duration:g} min by {mode}." if duration is not None else "Travel duration unknown."]
    metres, minutes, transfers = route.walking_distance_m.value, route.walking_time_min.value, route.transfers.value
    if metres is None:
        lines.append("Walking distance unknown.")
    else:
        estimate = TravelEstimate(mode=route.mode, source="onemap", confidence=0.5, walking_distance_m=metres)
        lines.append("Walking distance: " + ("0 m" if metres == 0 else estimate.walking_distance_display) + ".")
    lines.append("Walking time unknown." if minutes is None else f"Approximate walking time: {minutes:g} min.")
    lines.append("Transfer count unknown." if transfers is None else f"Known transfers: {transfers:g}.")
    return lines


def _render(context, rows, plan=None, *, fallback=False, directions=False):
    centres = {item.centre_id: item for item in context.centres}
    names = {item["id"]: _label(item["name"]) for item in load_hawker_centres()}
    order = ([centre_id for centre_id in context.approved_centre_ids or ()
              if centre_id in centres and centres[centre_id].primary_route is not None]
             if directions else _centre_order(context, rows))
    if not order:
        return "No eligible current route information is available." if directions else _NO_FOOD
    sources = {"model_ranked": "Location model-ranked selection.",
               "deterministic_fallback": "Location deterministic ranking fallback.",
               "menu_supplement": "Deterministic menu supplement; not ranked by the Location model.",
               "route_facts_only": "Verified route facts only; no accepted Location model ranking.",
               "catalogue_only": "Catalogue-only selection; no fresh Location ranking."}
    choice_ids = {choice.menu_item_id for choice in plan.choices} if plan is not None else set()
    lines = ["Verified route information:" if directions else "Eligible food options in verified Location order:"]
    if fallback:
        lines.append("Showing deterministic eligible selections; the food ranking was unavailable or unusable.")
    for index, centre_id in enumerate(order, 1):
        centre = centres.get(centre_id)
        if centre is None:
            continue
        lines.extend(["", f"{index}. {names.get(centre_id, centre_id)}", *["   " + text for text in _route_lines(centre)]])
        lines.append("   " + sources[centre.selection_source])
        lines.append("   Location rationale: " + centre.rationale)
        for row in rows:  # Preserve conflict-ranked order within each centre.
            if row["hawker_centre_id"] != centre_id or row["menu_item_id"] not in choice_ids:
                continue
            demo = "Illustrative demo listing and price: " if row.get("is_mock") is True else "Listed menu item: "
            unit = _label(row.get("unit_number") or row.get("unit"))
            price = _label(row.get("price_label")) or "price unavailable"
            lines.append(f"   {demo}{_label(row.get('stall_name'))}{' (' + unit + ')' if unit else ''} — {_label(row.get('menu_item_name'))}; listed price (SGD): {price}.")
            choice = next(item for item in plan.choices if item.menu_item_id == row["menu_item_id"])
            if "listed_within_budget" in choice.reason_refs:
                lines.append("   Listed item prices meet the requested budget.")
            if "starting_price_within_budget" in choice.reason_refs:
                lines.append("   Starting listed price meets the budget; other options may cost more.")
            if row.get("queue_minutes") is not None:
                lines.append(f"   Simulated queue estimate: about {row['queue_minutes']:g} min; not live conditions.")
    basis = ("Location confidence is model-reported and uncalibrated." if context.confidence_basis == "model_reported_uncalibrated"
             else "Location confidence is a policy value, not calibrated accuracy.")
    notes = [basis, *context.limitations]
    lines.extend(["", "Notes: " + " ".join(dict.fromkeys(notes))])
    return "\n".join(lines)
