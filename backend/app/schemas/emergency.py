"""
Emergency Response and Email Status Schemas (Phase 5B).
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict


class EmergencyStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    state: str
    priority: str
    reason: str
    fire_detection_available: bool
    raw_fire_detected: Optional[bool] = None
    fire_confidence: Optional[float] = None
    confirmation_threshold: float
    temporal_hits: int
    people_detection_available: bool
    visible_people_count: Optional[int] = None
    visual_smoke_detected: Optional[bool] = None
    visual_smoke_confidence: Optional[float] = None
    sensor_context_available: bool
    sensor_input_source: Optional[str] = None
    confirmation_source: Optional[str] = None
    observation_id: Optional[int] = None
    observation_timestamp: Optional[float] = None
    last_updated_at: str


class EmailStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    enabled: bool
    configured: bool
    last_attempt_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
