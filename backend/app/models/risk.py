"""
Risk assessment log ORM model.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text, Boolean
from app.database.session import Base


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sensor_reading_id = Column(Integer, ForeignKey("sensor_readings.id"), nullable=True)
    overall_risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    temp_risk = Column(Float, nullable=False)
    smoke_risk = Column(Float, nullable=False)
    gas_risk = Column(Float, nullable=False)
    flame_risk = Column(Float, nullable=False)
    camera_fire_risk = Column(Float, nullable=True)
    fire_detection_available = Column(Boolean, nullable=False, default=False)
    fire_confidence = Column(Float, nullable=True)
    visual_smoke_detected = Column(Boolean, nullable=True)
    visual_smoke_confidence = Column(Float, nullable=True)
    people_detected = Column(Integer, nullable=False, default=0)
    people_detection_available = Column(Boolean, nullable=False, default=False)
    emergency_priority = Column(String(50), nullable=False)
    contributing_factors = Column(Text, nullable=False)  # JSON-encoded array or newline list
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
