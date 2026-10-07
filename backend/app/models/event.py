"""
DetectionEvent ORM model for logging state-changes and notable detection events.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from app.database.session import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    fire_detection_available = Column(Boolean, nullable=False, default=False)
    fire_detected = Column(Boolean, nullable=False, default=False)
    fire_confidence = Column(Float, nullable=True)
    people_count = Column(Integer, nullable=False, default=0)
    people_detection_available = Column(Boolean, nullable=False, default=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    camera_status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
