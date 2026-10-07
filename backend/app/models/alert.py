"""
Alert ORM model with causal traceability to SensorReading, RiskAssessment, and Hazard Signature.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from app.database.session import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hazard_signature = Column(String(50), nullable=True, index=True)
    sensor_reading_id = Column(Integer, ForeignKey("sensor_readings.id"), nullable=True)
    risk_assessment_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)  # INFO, CAUTION, WARNING, HIGH, CRITICAL
    message = Column(String(255), nullable=False)
    people_detected = Column(Integer, nullable=False, default=0)
    people_detection_available = Column(Boolean, nullable=False, default=False)
    fire_detection_available = Column(Boolean, nullable=False, default=False)
    fire_detected = Column(Boolean, nullable=False, default=False)
    fire_confidence = Column(Float, nullable=True)
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
