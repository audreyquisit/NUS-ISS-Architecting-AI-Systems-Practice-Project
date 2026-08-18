from typing import Optional

from pydantic import BaseModel


class HawkerRequest(BaseModel):
    location: str
    dietary_preference: Optional[str] = None
    budget: float
    available_time_minutes: int
    weather: Optional[str] = None
