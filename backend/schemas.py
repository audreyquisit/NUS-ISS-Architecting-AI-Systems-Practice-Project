from pydantic import BaseModel, Field
from typing import List, Optional


class UserPreferences(BaseModel):
    location: str = Field(
        ..., description="Preferred location or hawker centre")
    dietary_preferences: Optional[List[str]] = Field(default_factory=list)
    budget: str = Field(
        ..., description="Budget category: low, medium, or high")
    available_time_min: int = Field(..., ge=0,
                                    description="Available time in minutes")
    weather: Optional[str] = Field(
        None, description="Current weather condition")
    max_queue: Optional[int] = Field(
        None, ge=0, description="Maximum acceptable queue length")


class StallRecommendation(BaseModel):
    stall_name: str
    hawker_centre: str
    cuisine: str
    price_range: str
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
