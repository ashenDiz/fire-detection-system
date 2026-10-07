"""
Sensor reading ORM model.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime
from app.database.session import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    flame_level = Column(Float, nullable=False)
    smoke_level = Column(Float, nullable=False)
    gas_level = Column(Float, nullable=False)
    input_source = Column(String(50), nullable=False, default="manual_simulation")
    scenario_tag = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
