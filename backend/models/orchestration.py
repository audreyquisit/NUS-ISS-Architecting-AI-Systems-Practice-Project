"""Structured planner contracts shared by the orchestrator and its workers."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


WorkerName = Literal["location", "dietary", "budget", "queue", "weather"]


class ParsedRequest(BaseModel):
    intent: Literal[
        "food_discovery", "centre_details", "directions", "smalltalk", "other"
    ]
    origin_text: Optional[str] = None
    target_centre_text: Optional[str] = None
    requested_area: Optional[Literal[
        "north", "north_east", "east", "central", "west", "south"
    ]] = None
    travel_mode: Optional[Literal["pt", "walk", "drive", "cycle"]] = None
    dietary_preferences: list[str] = Field(default_factory=list)
    cuisine_preferences: list[str] = Field(default_factory=list)
    menu_item_preferences: list[str] = Field(default_factory=list)
    excluded_menu_item_ids: list[str] = Field(default_factory=list)
    stall_name_preferences: list[str] = Field(default_factory=list)
    budget_amount: Optional[float] = Field(default=None, ge=0)
    available_time_min: Optional[int] = Field(default=None, ge=0)
    max_travel_time_min: Optional[int] = Field(default=None, ge=0)
    max_queue_min: Optional[int] = Field(default=None, ge=0)
    time_period: Optional[str] = None
    priorities: list[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    # The planner may phrase these labels with spaces despite the prompt's
    # underscore schema; normalize after parsing instead of rejecting the turn.
    field_sources: dict[str, str] = Field(default_factory=dict)
    uncertain_fields: list[str] = Field(default_factory=list)


class PlannedTask(BaseModel):
    agent: WorkerName
    task: str


class OrchestrationPlan(BaseModel):
    request: ParsedRequest
    tasks: list[PlannedTask] = Field(default_factory=list)


class AgentResult(BaseModel):
    agent: str
    status: Literal["ok", "partial", "needs_input", "unavailable", "error"] = "ok"
    candidates: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    location_required: bool = False
    payload: dict = Field(default_factory=dict)
