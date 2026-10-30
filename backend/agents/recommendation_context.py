"""Immutable verifier-owned projection, kept only in private graph state.

This is data for a future Recommendation boundary, not a replacement for the
Location grounding bundle or an authority that may be restored from client JSON.
Normal routed selections remain bounded by three; catalogue-only projections
allow up to 256 centres (the current catalogue has 123). Public results are never
truncated by this projection. Food ownership follows the existing 30-item join.
"""

from typing import Annotated, Literal, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field, StrictInt


_MAX_CENTRES = 256
_Id = Annotated[str, Field(min_length=1, max_length=160)]
_Text = Annotated[str, Field(max_length=1600)]
_Limit = Annotated[str, Field(max_length=512)]
_Value = Union[StrictInt, Annotated[float, Field(strict=True, allow_inf_nan=False)], None]
_Source = Literal["model_ranked", "deterministic_fallback", "menu_supplement", "route_facts_only", "catalogue_only"]
_FitBasis = Literal["model_reported_uncalibrated", "deterministic_policy", "not_location_ranked", "no_model_fit"]


class _PrivateModel(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}


class _Measurement(_PrivateModel):
    value: _Value
    provenance: Literal["measured", "derived", "unavailable"]


class _VerifiedPrimaryRoute(_PrivateModel):
    source: Literal["onemap"] = "onemap"
    mode: Literal["pt", "walk", "drive", "cycle"]
    total_duration_min: _Measurement
    walking_distance_m: _Measurement
    walking_time_min: _Measurement
    transfers: _Measurement


class _VerifiedCentre(_PrivateModel):
    centre_id: _Id
    rank: int = Field(ge=1)
    rationale: _Text
    logistics_fit: float = Field(ge=0, le=1, allow_inf_nan=False)
    fit_basis: _FitBasis
    selection_source: _Source
    primary_route: Optional[_VerifiedPrimaryRoute] = None


class _VerifiedLocationContext(_PrivateModel):
    centres: Tuple[_VerifiedCentre, ...] = Field(max_length=_MAX_CENTRES)
    reasoning: _Text
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence_basis: Literal["model_reported_uncalibrated", "deterministic_policy"]
    limitations: Tuple[_Limit, ...] = Field(max_length=16)
    truncated: bool = False


class _EligibleMenuItem(_PrivateModel):
    menu_item_id: _Id
    centre_id: _Id
    stall_id: _Id


class _RecommendationContext(_PrivateModel):
    context_id: str = Field(default_factory=lambda: uuid4().hex, min_length=32, max_length=32)
    request_id: Optional[str] = Field(default=None, max_length=160)
    # None means Location scope is not applicable; () is applicable empty approval.
    approved_centre_ids: Optional[Tuple[_Id, ...]] = Field(default=None, max_length=_MAX_CENTRES)
    centres: Tuple[_VerifiedCentre, ...] = Field(default=(), max_length=_MAX_CENTRES)
    location_reasoning: _Text = "Location scope is not applicable."
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    confidence_basis: Literal["model_reported_uncalibrated", "deterministic_policy", "not_applicable"] = "not_applicable"
    limitations: Tuple[_Limit, ...] = Field(default=(), max_length=24)
    eligible_menu_items: Tuple[_EligibleMenuItem, ...] = Field(default=(), max_length=30)
    truncated: bool = False


def _project_primary_route(travel: dict) -> _VerifiedPrimaryRoute:
    """Input is the verifier's sanitized primary route, never provider prose."""
    def measurement(metric):
        value = travel.get(metric)
        provenance = ("unavailable" if value is None else
                      "derived" if metric == "walking_time_min" else "measured")
        return _Measurement(value=value, provenance=provenance)

    return _VerifiedPrimaryRoute(mode=travel["mode"], **{
        metric: measurement(metric) for metric in (
            "total_duration_min", "walking_distance_m", "walking_time_min", "transfers",
        )
    })


def _build_recommendation_context(*, request_id, approved_centres, locations, joined, food_requested,
                                  dietary_requested=False, budget_requested=False) -> _RecommendationContext:
    """Project only accepted verifier data. No parsing of model/worker prose.

    Multiple Location envelopes share only their approval intersection. If a
    centre's accepted descriptions disagree, retain identity without elevating
    any one envelope's route, fit, confidence or ranking provenance.
    """
    centres = []
    truncated = any(location.truncated for location in locations)
    limitations = ["Menu catalogue coverage is partial."] if food_requested else []
    if dietary_requested:
        limitations.append("Catalogue dietary matching may use inference; ingredient, allergen and certification evidence is not established.")
    if budget_requested:
        limitations.append("Budget eligibility concerns listed menu items, not a complete meal or a live price guarantee.")
    if locations:
        for centre in locations[0].centres:
            if approved_centres is not None and centre.centre_id not in approved_centres:
                continue
            copies = [next((item for item in location.centres if item.centre_id == centre.centre_id), None)
                      for location in locations]
            if any(item != centre for item in copies):
                centre = _VerifiedCentre(centre_id=centre.centre_id, rank=len(centres) + 1,
                    rationale="Catalogue identity is approved; conflicting Location descriptions were not retained.",
                    logistics_fit=0, fit_basis="no_model_fit", selection_source="catalogue_only")
            centres.append(centre)
    approved = None if approved_centres is None else tuple(item.centre_id for item in centres)
    if len(locations) == 1:
        location = locations[0]
        reasoning, confidence, basis = location.reasoning, location.confidence, location.confidence_basis
        limitations.extend(location.limitations)
    elif approved_centres is not None:
        reasoning = "Only the common approved Location scope is retained; no single ranking confidence is asserted."
        confidence, basis = (0.5 if centres else 0.0), "deterministic_policy"
        limitations.append("Confidence is a conservative policy value, not calibrated accuracy.")
    else:
        reasoning, confidence, basis = "Location scope is not applicable.", None, "not_applicable"
    if len(locations) > 1:
        limitations.append("Multiple Location results: conflicting per-centre descriptions lose route and model-fit authority.")
    if truncated:
        limitations.append("Private catalogue projection is bounded to 256 centres; omitted centres carry no private recommendation authority.")
    menu = []
    seen = set()
    for item in joined[:30]:
        if approved is not None and item["hawker_centre_id"] not in approved:
            continue
        if item["menu_item_id"] in seen:
            continue
        seen.add(item["menu_item_id"])
        menu.append(_EligibleMenuItem(menu_item_id=item["menu_item_id"],
            centre_id=item["hawker_centre_id"], stall_id=item["stall_id"]))
    if food_requested and not menu:
        limitations.append("No eligible joined menu items remain; catalogue coverage is not exhaustive.")
    if any(item.get("is_mock") is True for item in joined):
        limitations.append("Mock menu listings and prices are illustrative demo data.")
    if any(item.get("queue_source") == "synthetic_queue_mock" for item in joined):
        limitations.append("Joined queue estimates are simulated demo data, not live conditions.")
    return _RecommendationContext(request_id=request_id, approved_centre_ids=approved,
        centres=tuple(centres), location_reasoning=reasoning, confidence=confidence,
        confidence_basis=basis, limitations=tuple(dict.fromkeys(limitations)),
        eligible_menu_items=tuple(menu), truncated=truncated)
