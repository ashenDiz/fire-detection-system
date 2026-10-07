"""
Sensor Data Provider abstraction and Manual Provider implementation.
Designed for seamless future expansion with ESP32SensorProvider or ArduinoSensorProvider.
"""

from abc import ABC, abstractmethod
from typing import Optional
from sqlalchemy.orm import Session

from app.models.sensor import SensorReading
from app.schemas.sensor import SensorReadingCreate


class SensorDataProvider(ABC):
    """Abstract Base Class for environmental sensor telemetry providers."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Identifier for the sensor data provider."""
        pass

    @abstractmethod
    def get_sensor_mode_label(self) -> str:
        """Human-readable display label for the dashboard."""
        pass

    @abstractmethod
    def record_reading(self, reading_data: SensorReadingCreate, db: Session) -> SensorReading:
        """Persist an incoming sensor reading to the database."""
        pass

    @abstractmethod
    def get_latest_reading(self, db: Session) -> Optional[SensorReading]:
        """Fetch the most recent sensor reading from persistence."""
        pass


class ManualSensorProvider(SensorDataProvider):
    """
    Sensor provider for manual simulation and research testing.
    All inputs are strictly tagged with input_source='manual_simulation'.
    """

    def get_provider_name(self) -> str:
        return "manual_simulation"

    def get_sensor_mode_label(self) -> str:
        return "Manual Simulation"

    def record_reading(self, reading_data: SensorReadingCreate, db: Session) -> SensorReading:
        db_reading = SensorReading(
            temperature=reading_data.temperature,
            humidity=reading_data.humidity,
            flame_level=reading_data.flame_level,
            smoke_level=reading_data.smoke_level,
            gas_level=reading_data.gas_level,
            input_source="manual_simulation",
            scenario_tag=reading_data.scenario_tag
        )
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)
        return db_reading

    def get_latest_reading(self, db: Session) -> Optional[SensorReading]:
        return db.query(SensorReading).order_by(SensorReading.created_at.desc()).first()


# Global provider instance for Phase 1
active_sensor_provider: SensorDataProvider = ManualSensorProvider()


def get_sensor_provider() -> SensorDataProvider:
    """Dependency helper to retrieve the active sensor provider."""
    return active_sensor_provider
