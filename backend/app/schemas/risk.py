"""
Pydantic schemas for risk assessment outputs, alerts, and detection events.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from app.schemas.sensor import SensorReadingResponse


class RiskAssessmentResponse(BaseModel):
    id: Optional[int] = None
    sensor_reading_id: Optional[int] = None
    overall_risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str  # SAFE, CAUTION, WARNING, HIGH RISK, CRITICAL
    temp_risk: float
    smoke_risk: float
    gas_risk: float
    flame_risk: float
    camera_fire_risk: Optional[float] = None
    fire_detection_available: bool = False
    camera_fire_confidence: Optional[float] = None
    fire_confidence: Optional[float] = None
    visual_smoke_detected: Optional[bool] = None
    visual_smoke_confidence: Optional[float] = None
    people_detected: Optional[int] = None
    people_detection_available: bool = False
    emergency_priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    contributing_factors: List[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("contributing_factors", mode="before")
    def parse_factors(cls, v):
        if isinstance(v, str):
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return [str(parsed)]
            except Exception:
                return [v] if v else []
        return v

    @model_validator(mode="after")
    def enforce_availability(self):
        if not self.people_detection_available:
            self.people_detected = None
        if not self.fire_detection_available:
            self.camera_fire_risk = None
            self.camera_fire_confidence = None
            self.fire_confidence = None
            self.visual_smoke_detected = None
            self.visual_smoke_confidence = None
        else:
            if self.camera_fire_confidence is None and self.fire_confidence is not None:
                self.camera_fire_confidence = self.fire_confidence
            elif self.fire_confidence is None and self.camera_fire_confidence is not None:
                self.fire_confidence = self.camera_fire_confidence
        return self


class AlertResponse(BaseModel):
    id: int
    hazard_signature: Optional[str] = None
    sensor_reading_id: Optional[int] = None
    risk_assessment_id: Optional[int] = None
    risk_score: float
    risk_level: str
    alert_type: str
    message: str
    people_detected: Optional[int] = None
    people_detection_available: bool = False
    fire_detection_available: bool = False
    fire_detected: Optional[bool] = None
    fire_confidence: Optional[float] = None
    acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def enforce_availability(self):
        if not self.people_detection_available:
            self.people_detected = None
        if not self.fire_detection_available:
            self.fire_detected = None
            self.fire_confidence = None
        return self


class DetectionEventResponse(BaseModel):
    id: int
    fire_detection_available: bool = False
    fire_detected: Optional[bool] = None
    fire_confidence: Optional[float] = None
    people_count: Optional[int] = None
    people_detection_available: bool = False
    risk_score: float
    risk_level: str
    camera_status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def enforce_availability(self):
        if not self.people_detection_available:
            self.people_count = None
        if not self.fire_detection_available:
            self.fire_detected = None
            self.fire_confidence = None
        return self


class SensorSubmissionResult(BaseModel):
    reading: SensorReadingResponse
    risk: RiskAssessmentResponse
    active_alert: Optional[AlertResponse] = None
