"""Typed request and result contracts for the Location & Logistics agent."""

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class LocationQuery(BaseModel):
    """Input from the orchestrator (or the direct location API)."""

    request: str = ""
    user_location: Optional[Coordinates] = None
    user_location_source: Optional[Literal[
        "browser_current_location", "provided_coordinates"
    ]] = None
    location_text: Optional[str] = None
    travel_mode: Optional[Literal["pt", "walk", "drive", "cycle"]] = None
    max_travel_time_min: Optional[int] = Field(default=None, ge=0)
    priorities: List[str] = Field(default_factory=list)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    parsed_request: Optional[Dict[str, Any]] = None


class TravelEstimate(BaseModel):
    total_duration_min: Optional[int] = Field(default=None, ge=0)
    walking_distance_m: Optional[float] = Field(default=None, ge=0)
    walking_distance_display: Optional[str] = None
    walking_time_min: Optional[int] = Field(default=None, ge=0)
    walking_time_display: Optional[str] = None
    mode: Literal["pt", "walk", "drive", "cycle"]
    source: Literal["onemap", "unavailable"]
    confidence: float = Field(ge=0, le=1)
    transfers: Optional[int] = Field(default=None, ge=0)
    limitation: Optional[str] = None
    itinerary_summary: Optional[str] = None

    @model_validator(mode="after")
    def add_walking_displays(self):
        metres = self.walking_distance_m
        if metres is None:
            return self

        if metres >= 900:
            distance = "about 1 km" if metres < 1000 else f"about {math.floor(metres / 100 + 0.5) / 10:.1f} km"
        else:
            rounded_metres = max(100, math.floor(metres / 100 + 0.5) * 100)
            distance = f"about {rounded_metres} m" if metres >= 50 else "under 100 m"
        self.walking_distance_display = self.walking_distance_display or distance

        # Approximate walking pace: 5 km/h (about 83 m/min), rounded to a
        # whole minute for a user-facing estimate.
        minutes = max(1, math.floor(metres * 60 / 5000 + 0.5)) if metres > 0 else 0
        self.walking_time_min = self.walking_time_min if self.walking_time_min is not None else minutes
        if self.walking_time_display is None:
            self.walking_time_display = f"about {minutes} min" if minutes else "under 1 min"
        return self


class CentreLogistics(BaseModel):
    centre_id: str
    centre_name: str
    address: str
    coordinates: Coordinates
    travel: TravelEstimate
    alternative_travel: Optional[TravelEstimate] = None
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class RankedCentre(BaseModel):
    centre_id: str
    rank: int = Field(ge=1)
    logistics_fit: float = Field(ge=0, le=1)
    rationale: str


class LocationDecision(BaseModel):
    selected_centres: List[RankedCentre]
    logistics: List[CentreLogistics]
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    limitations: List[str] = Field(default_factory=list)
    origin: Optional[Coordinates] = None
    origin_label: Optional[str] = None
    origin_source: Optional[Literal[
        "free_text_place", "browser_current_location", "provided_coordinates"
    ]] = None
    intent_interpretation: Dict[str, Any] = Field(default_factory=dict)
