from typing import List, Optional

from pydantic import BaseModel


class HawkerCandidate(BaseModel):
    stall_name: str
    hawker_centre: str

    distance_minutes: Optional[int] = None
    price: Optional[float] = None
    queue_minutes: Optional[int] = None

    dietary_suitable: Optional[bool] = None
    weather_suitable: Optional[bool] = None


class AgentResult(BaseModel):
    agent: str

    candidates: List[HawkerCandidate] = []

    reasoning: Optional[str] = None

    verified: bool = False