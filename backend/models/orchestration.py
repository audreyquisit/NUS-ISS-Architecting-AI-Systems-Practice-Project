"""Structured planner contracts shared by the orchestrator and its workers."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


WorkerName = Literal["location", "dietary", "budget", "queue", "weather"]


class ParsedRequest(BaseModel):
    intent: Literal[
        "food_discovery", "centre_details", "directions", "smalltalk", "other",
        "queue",
    ]
    origin_text: Optional[str] = Field(default=None, max_length=200)
    target_centre_text: Optional[str] = Field(default=None, max_length=200)
    requested_area: Optional[str] = Field(default=None, max_length=80)
    search_scope_mode: Optional[Literal["nearby", "region", "global"]] = None
    travel_mode: Optional[Literal["pt", "walk", "drive", "cycle"]] = None
    dietary_preferences: list[str] = Field(default_factory=list, max_length=12)
    cuisine_preferences: list[str] = Field(default_factory=list, max_length=12)
    menu_item_preferences: list[str] = Field(default_factory=list, max_length=20)
    excluded_menu_item_ids: list[str] = Field(default_factory=list, max_length=100)
    stall_name_preferences: list[str] = Field(default_factory=list, max_length=20)
    budget_amount: Optional[float] = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    available_time_min: Optional[int] = Field(default=None, ge=0, le=1440)
    max_travel_time_min: Optional[int] = Field(default=None, ge=0, le=1440)
    max_queue_min: Optional[int] = Field(default=None, ge=0, le=1440)
    time_period: Optional[str] = Field(default=None, max_length=80)
    priorities: list[str] = Field(default_factory=list, max_length=12)
    clarification_question: Optional[str] = Field(default=None, max_length=400)
    # The planner may phrase these labels with spaces despite the prompt's
    # underscore schema; normalize after parsing instead of rejecting the turn.
    field_sources: dict[str, Any] = Field(default_factory=dict)
    uncertain_fields: list[str] = Field(default_factory=list)


class PlannedTask(BaseModel):
    agent: WorkerName
    task: str = Field(max_length=500)


class OrchestrationPlan(BaseModel):
    request: ParsedRequest
    tasks: list[PlannedTask] = Field(default_factory=list, max_length=10)


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
