"""Private, request-local Location evidence. Never serialize into public results.

Snapshots bind accepted claims to primary evidence; their digests check consistency,
not authenticity. Only the fresh Location execution creates the authoritative bundle.
"""

import hashlib
import json
import math
from typing import Annotated, Literal, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field, FiniteFloat, StrictInt, ValidationError

from models.location_models import TravelEstimate


_Metric = Literal["total_duration_min", "walking_distance_m", "transfers", "mode", "walking_time_min"]
_METRICS = ("total_duration_min", "walking_distance_m", "transfers", "mode", "walking_time_min")
_Source = Literal["grounded_model", "deterministic_fallback", "menu_expansion"]
_Basis = Literal["model_reported_uncalibrated", "deterministic_policy", "not_location_ranked"]
_CentreId = Annotated[str, Field(max_length=160)]
_FactId = Annotated[str, Field(max_length=240)]


class _PrivateModel(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}


class _Fact(_PrivateModel):
    id: str = Field(max_length=240)
    metric: _Metric
    value: Union[StrictInt, FiniteFloat, Literal["pt", "walk", "drive", "cycle"], None]
    provenance: Literal["measured", "derived", "unavailable"]


class _Reason(_PrivateModel):
    code: Literal[
        "PRIMARY_ROUTE", "SHORTER_DURATION", "LESS_MEASURED_WALKING",
        "FEWER_KNOWN_TRANSFERS", "BALANCED_TRADEOFF", "METRIC_UNAVAILABLE",
        "DETERMINISTIC_ORDER",
    ]
    evidence_refs: Tuple[_FactId, ...] = Field(min_length=1, max_length=6)
    comparison_centre_id: Optional[str] = Field(default=None, max_length=160)


class _PrimaryRoute(_PrivateModel):
    centre_id: str = Field(max_length=160)
    route_id: str = Field(max_length=180)
    identity_digest: str = Field(min_length=64, max_length=64)
    source: Literal["onemap"] = "onemap"
    facts: Tuple[_Fact, ...] = Field(min_length=5, max_length=5)


class _Selection(_PrivateModel):
    centre_id: str = Field(max_length=160)
    route_id: str = Field(max_length=180)
    original_rank: Optional[int] = Field(default=None, ge=1, le=3)
    source: _Source
    logistics_fit: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    fit_basis: _Basis
    rationale_digest: Optional[str] = Field(default=None, min_length=64, max_length=64)
    reasons: Tuple[_Reason, ...] = Field(default=(), max_length=3)
    status: Literal["active", "superseded"] = "active"
    status_reason: Literal[
        "retained", "not_selected", "route_evidence_changed", "comparison_evidence_changed", "metadata_changed",
    ] = "retained"


class _Supplement(_PrivateModel):
    selection: _Selection
    route: _PrimaryRoute


class _LocationGrounding(_PrivateModel):
    version: Literal[1] = 1
    execution_id: str = Field(min_length=32, max_length=32)
    request_id: Optional[str] = Field(default=None, max_length=160)
    decision_source: Literal["grounded_model", "deterministic_fallback"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence_basis: Literal["model_reported_uncalibrated", "deterministic_policy"]
    fallback_reason: Optional[str] = Field(default=None, max_length=160)
    fallback_order: Tuple[_Metric, ...] = Field(default=(), max_length=3)
    max_travel_time_min: Optional[int] = Field(default=None, ge=0)
    routes: Tuple[_PrimaryRoute, ...] = Field(max_length=8)
    selections: Tuple[_Selection, ...] = Field(max_length=3)
    supplements: Tuple[_Supplement, ...] = Field(default=(), max_length=3)
    active_order: Tuple[_CentreId, ...] = Field(max_length=3)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _primary_identity(item: dict) -> dict:
    def canonical(value):
        # Preserve exact numeric values: never round or convert integers to floats.
        # bool stays a distinct JSON type; integral floats include signed zero.
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: canonical(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [canonical(nested) for nested in value]
        return value

    return canonical({key: item.get(key) for key in (
        "centre_id", "centre_name", "address", "coordinates", "route_id", "travel",
    )})


def _usable_primary_routes(supplied_routes) -> tuple[dict, bool]:
    """The Phase 5A policy, shared by expansion and private evidence matching."""
    primary_by_id, route_by_id, ambiguous_ids = {}, {}, set()
    excluded = False
    for item in supplied_routes:
        if not isinstance(item, dict) or not isinstance(item.get("centre_id"), str) or not item["centre_id"]:
            excluded = True
            continue
        centre_id = item["centre_id"]
        # Match snapshot fingerprints, including JSON's bool/number distinction.
        primary = _digest(_primary_identity(item))
        if centre_id in primary_by_id and primary_by_id[centre_id] != primary:
            ambiguous_ids.add(centre_id)
        else:
            primary_by_id.setdefault(centre_id, primary)
            route_by_id.setdefault(centre_id, item)
    usable = {}
    for centre_id, item in route_by_id.items():
        travel = item.get("travel")
        minutes = travel.get("total_duration_min") if isinstance(travel, dict) else None
        if (
            centre_id not in ambiguous_ids
            and isinstance(travel, dict) and travel.get("source") == "onemap"
            and isinstance(travel.get("mode"), str)
            and travel.get("mode") in {"pt", "walk", "drive", "cycle"}
            and not any(isinstance(travel.get(metric), bool) for metric in _METRICS)
            and isinstance(minutes, (int, float)) and not isinstance(minutes, bool)
            and minutes >= 0 and (isinstance(minutes, int) or math.isfinite(minutes))
        ):
            usable[centre_id] = item
        else:
            excluded = True
    return usable, excluded


def _primary_facts(centre_id: str, travel: dict) -> list[dict]:
    facts = []
    for metric in _METRICS:
        value = travel.get(metric)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            value = None
        provenance = ("unavailable" if value is None else
                      "derived" if metric == "walking_time_min" else "measured")
        facts.append({"id": f"{centre_id}:primary:{metric}", "metric": metric,
                      "value": value, "provenance": provenance})
    return facts


def _snapshot(item: dict, facts=None) -> _PrimaryRoute:
    centre_id = item["centre_id"]
    return _PrimaryRoute(
        centre_id=centre_id, route_id=f"{centre_id}:primary",
        identity_digest=_digest(_primary_identity(item)),
        facts=tuple(_Fact(**fact) for fact in (
            facts if facts is not None else _primary_facts(centre_id, item["travel"])
        )),
    )


def _capture_grounding(query, shortlist, ranking, choices=None, facts=None,
                       fallback_reason=None, fallback_order=()) -> _LocationGrounding:
    """Called only after existing selection validation or deterministic policy."""
    source = "deterministic_fallback" if fallback_reason is not None else "grounded_model"
    basis = "deterministic_policy" if fallback_reason is not None else "model_reported_uncalibrated"
    routes = tuple(_snapshot(item.model_dump(mode="json"), [
        {key: facts[f"{item.centre_id}:primary:{metric}"][key]
         for key in ("id", "metric", "value", "provenance")}
        for metric in _METRICS
    ] if facts is not None else None) for item in shortlist)
    selections = []
    for selected in ranking.selected_centres:
        reasons = (choices[selected.centre_id].reasons if choices is not None else [_Reason(
            code="DETERMINISTIC_ORDER", evidence_refs=tuple(
                f"{selected.centre_id}:primary:{metric}" for metric in fallback_order
            ),
        )])
        selections.append(_Selection(
            centre_id=selected.centre_id, route_id=f"{selected.centre_id}:primary",
            original_rank=selected.rank, source=source, logistics_fit=selected.logistics_fit,
            fit_basis=basis, rationale_digest=_digest(selected.rationale),
            reasons=tuple(_Reason(**reason.model_dump()) for reason in reasons),
        ))
    return _LocationGrounding(
        execution_id=uuid4().hex, request_id=query.request_id,
        decision_source=source, confidence=ranking.confidence, confidence_basis=basis,
        fallback_reason=fallback_reason, fallback_order=tuple(fallback_order),
        max_travel_time_min=query.max_travel_time_min, routes=routes,
        selections=tuple(selections), active_order=tuple(item.centre_id for item in selections),
    )


def _reconcile_grounding(bundle, decision: dict, request_id=None, *, allow_supplements=False):
    """Copy-on-update; never turn a cached dict or public prose into fresh proof."""
    if not isinstance(bundle, _LocationGrounding) or bundle.request_id != request_id:
        return None
    usable, _ = _usable_primary_routes(decision.get("logistics") or [])
    original_routes = {route.centre_id: route for route in bundle.routes}
    matched = {
        centre_id for centre_id, route in original_routes.items()
        if centre_id in usable and route.identity_digest == _digest(_primary_identity(usable[centre_id]))
    }
    final = {item["centre_id"]: item for item in (decision.get("selected_centres") or [])[:3]
             if isinstance(item, dict) and isinstance(item.get("centre_id"), str)}
    selections = []
    for selected in bundle.selections:
        centre_id = selected.centre_id
        reason = "retained"
        if selected.status == "superseded":
            reason = selected.status_reason
        elif centre_id not in matched:
            reason = "route_evidence_changed"
        elif centre_id not in final:
            reason = "not_selected"
        elif any(claim.comparison_centre_id is not None and claim.comparison_centre_id not in matched
                 for claim in selected.reasons):
            reason = "comparison_evidence_changed"
        elif (final[centre_id].get("logistics_fit") != selected.logistics_fit
              or _digest(final[centre_id].get("rationale")) != selected.rationale_digest):
            reason = "metadata_changed"
        selections.append(selected.model_copy(update={
            "status": "active" if reason == "retained" else "superseded", "status_reason": reason,
        }))
    original_ids = {item.centre_id for item in selections}
    previous_supplements = {item.selection.centre_id: item for item in bundle.supplements}
    supplements = []
    for centre_id in final:
        if centre_id in original_ids or centre_id not in usable:
            continue
        minutes = usable[centre_id]["travel"]["total_duration_min"]
        if bundle.max_travel_time_min is not None and minutes > bundle.max_travel_time_min:
            continue
        previous = previous_supplements.get(centre_id)
        if not allow_supplements and previous is None:
            continue
        try:
            route = _snapshot(usable[centre_id])
        except ValidationError:
            # A malformed external evidence record cannot become a supplement.
            continue
        if not allow_supplements and previous.route != route:
            continue
        supplements.append(_Supplement(
            route=route, selection=_Selection(
                centre_id=centre_id, route_id=route.route_id,
                source="menu_expansion", fit_basis="not_location_ranked",
            ),
        ))
    active = {item.centre_id for item in selections if item.status == "active"}
    active.update(item.selection.centre_id for item in supplements)
    return bundle.model_copy(update={
        "selections": tuple(selections), "supplements": tuple(supplements),
        "active_order": tuple(centre_id for centre_id in final if centre_id in active),
    })


class _RankingOutputError(ValueError):
    """An unusable ranking response, distinct from application ValueError."""


_METRIC_LABELS = {
    "total_duration_min": "shortest travel duration",
    "walking_distance_m": "least measured walking distance",
    "transfers": "fewest measured transfers",
}


def _render_route_reason(centre_id, reason, names, facts, objectives=()):
    """Shared factual validation/rendering; preference interpretation stays in Location."""
    def metric_text(metric, value):
        if metric == "total_duration_min":
            return f"{value} min"
        if metric == "transfers":
            return f"{value} transfer(s)"
        estimate = TravelEstimate(
            mode="pt", source="onemap", confidence=0.5, walking_distance_m=value,
        )
        return f"{estimate.walking_distance_display} walking"

    refs = reason.evidence_refs
    if len(refs) != len(set(refs)) or any(ref not in facts for ref in refs):
        raise _RankingOutputError("invalid_evidence_reference")
    other_id = reason.comparison_centre_id
    other = names.get(other_id)
    if other_id is not None and (other is None or other_id == centre_id):
        raise _RankingOutputError("invalid_comparison_centre")
    def ref(centre_id, metric):
        return f"{centre_id}:primary:{metric}"
    def pair(metric):
        expected = {ref(centre_id, metric), ref(other_id, metric)}
        if not expected <= set(refs):
            raise _RankingOutputError("missing_comparison_evidence")
        values = [facts[key] for key in (ref(centre_id, metric), ref(other_id, metric))]
        if any(fact["provenance"] != "measured" or fact["value"] is None for fact in values):
            raise _RankingOutputError("unavailable_comparison_metric")
        return [fact["value"] for fact in values]
    if reason.code == "PRIMARY_ROUTE":
        if other_id is not None or set(refs) != {ref(centre_id, "total_duration_min")}:
            raise _RankingOutputError("invalid_primary_route_reason")
        return f"Verified primary route: {facts[ref(centre_id, 'total_duration_min')]['value']} min."
    if reason.code == "METRIC_UNAVAILABLE":
        if other_id is not None or any(
            facts[key]["centre_id"] != centre_id
            or facts[key]["metric"] not in {"walking_distance_m", "transfers", "walking_time_min"}
            or facts[key]["provenance"] != "unavailable" for key in refs
        ):
            raise _RankingOutputError("invalid_missing_metric_reason")
        names = {"walking_distance_m": "Walking distance", "transfers": "Transfer count",
                 "walking_time_min": "Walking-time estimate"}
        return "; ".join(f"{names[facts[key]['metric']]} unavailable" for key in refs) + "."
    if other is None:
        raise _RankingOutputError("missing_comparison_centre")
    if reason.code == "BALANCED_TRADEOFF":
        metrics = ["total_duration_min"]
        for metric in ("walking_distance_m", "transfers"):
            if ref(centre_id, metric) in refs or ref(other_id, metric) in refs:
                metrics.append(metric)
        if len(metrics) < 2 or set(refs) != {
            ref(centre, metric) for centre in (centre_id, other_id) for metric in metrics
        }:
            raise _RankingOutputError("invalid_tradeoff_evidence")
        values = [pair(metric) for metric in metrics]
        if not any(a < b for a, b in values) or not any(a > b for a, b in values):
            raise _RankingOutputError("not_a_measured_tradeoff")
        if objectives and not any(
            a < b and metric in objectives for metric, (a, b) in zip(metrics, values)
        ):
            raise _RankingOutputError("tradeoff_conflicts_with_objectives")
        return "Trade-off versus " + other + ": " + "; ".join(
            f"{metric_text(metric, a)} versus {metric_text(metric, b)}"
            for metric, (a, b) in zip(metrics, values)
        ) + "."
    metric = {"SHORTER_DURATION": "total_duration_min", "LESS_MEASURED_WALKING": "walking_distance_m",
              "FEWER_KNOWN_TRANSFERS": "transfers"}[reason.code]
    if set(refs) != {ref(centre_id, metric), ref(other_id, metric)}:
        raise _RankingOutputError("invalid_comparison_evidence")
    a, b = pair(metric)
    if a >= b:
        raise _RankingOutputError("false_comparative_claim")
    return f"{metric_text(metric, a)} versus {other}'s {metric_text(metric, b)}."


def _matching_snapshot(snapshot, item: dict) -> bool:
    """Check both identity and captured facts, without trusting a digest alone."""
    try:
        expected = _snapshot(item)
    except ValidationError:
        return False
    return (
        snapshot.centre_id == expected.centre_id and snapshot.route_id == expected.route_id
        and snapshot.source == expected.source and snapshot.identity_digest == expected.identity_digest
        and len(snapshot.facts) == len(expected.facts)
        and all(
            not isinstance(actual.value, bool)
            and actual.id == fact.id and actual.metric == fact.metric
            and actual.value == fact.value and actual.provenance == fact.provenance
            for actual, fact in zip(snapshot.facts, expected.facts)
        )
    )


def _render_verified_choice(bundle, choice, matched, names) -> str:
    """Render accepted private claims only against independently matched snapshots."""
    basis = ("deterministic_policy" if choice.source == "deterministic_fallback"
             else "model_reported_uncalibrated")
    if (choice.status != "active" or choice.centre_id not in matched
            or choice.route_id != matched[choice.centre_id].route_id
            or choice.source != bundle.decision_source or choice.fit_basis != basis
            or bundle.confidence_basis != basis or not choice.reasons
            or not isinstance(choice.original_rank, int) or isinstance(choice.original_rank, bool)
            or not isinstance(choice.logistics_fit, (int, float)) or isinstance(choice.logistics_fit, bool)
            or not math.isfinite(choice.logistics_fit) or not 0 <= choice.logistics_fit <= 1):
        raise _RankingOutputError("invalid_selection_provenance")
    facts = {
        fact.id: {**fact.model_dump(), "centre_id": route.centre_id, "route_id": route.route_id}
        for route in matched.values() for fact in route.facts
    }
    if choice.source == "deterministic_fallback":
        order = bundle.fallback_order
        if (not bundle.fallback_reason or not order or len(order) != len(set(order))
                or "total_duration_min" not in order or any(metric not in _METRIC_LABELS for metric in order)
                or choice.logistics_fit != 0.5 or bundle.confidence != 0.5
                or len(choice.reasons) != 1 or choice.reasons[0].code != "DETERMINISTIC_ORDER"
                or choice.reasons[0].comparison_centre_id is not None
                or choice.reasons[0].evidence_refs != tuple(
                    f"{choice.centre_id}:primary:{metric}" for metric in order)):
            raise _RankingOutputError("invalid_fallback_provenance")
        policy = ", then ".join(_METRIC_LABELS[metric] for metric in order)
        minutes = facts[f"{choice.centre_id}:primary:total_duration_min"]["value"]
        return f"Deterministic fallback ordered by {policy}; OneMap travel duration is {minutes} min."
    if bundle.fallback_reason is not None or bundle.fallback_order:
        raise _RankingOutputError("invalid_model_provenance")
    if any(reason.code == "DETERMINISTIC_ORDER" for reason in choice.reasons):
        raise _RankingOutputError("invalid_model_reason")
    return " ".join(_render_route_reason(choice.centre_id, reason, names, facts)
                    for reason in choice.reasons)
