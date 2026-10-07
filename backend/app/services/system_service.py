"""
System Diagnostic and Transparency Service.
Monitors the state of backend, database, camera, AI models, and sensor telemetry mode.
Strictly adheres to research transparency: never reports fake online status or fake models.
"""

from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.system import SystemStatusResponse
from app.services.sensor_provider import get_sensor_provider
from app.services.camera_service import CameraService


class SystemService:
    @classmethod
    def get_status(cls, db: Session) -> SystemStatusResponse:
        # Check database connectivity
        try:
            db.execute(text("SELECT 1"))
            db_status = "Connected"
        except Exception as e:
            db_status = f"Error: {str(e)}"

        # Check fire detector model readiness directly from FireDetectionService (single source of truth)
        try:
            from app.services.fire_detection_service import FireDetectionService
            if FireDetectionService.is_model_loaded():
                fire_model_status = "Model Loaded & Initialized"
            else:
                raw_status = FireDetectionService.get_model_status()
                if raw_status and raw_status.startswith("Error"):
                    fire_model_status = raw_status
                else:
                    fire_model_status = "Fire AI Model Not Loaded"
        except Exception as e:
            fire_model_status = f"Error: {str(e)}"

        # Check person detector model readiness (Phase 3)
        try:
            from app.services.person_detection_service import PersonDetectionService
            person_model_status = PersonDetectionService.get_model_status()
        except Exception:
            person_model_status = "Not Loaded"

        # Camera connection status from independent CameraService
        camera_status = CameraService.get_status()

        # Sensor provider label
        sensor_provider = get_sensor_provider()
        sensor_mode = sensor_provider.get_sensor_mode_label()

        return SystemStatusResponse(
            backend="Online",
            database=db_status,
            camera=camera_status,
            person_model=person_model_status,
            fire_model=fire_model_status,
            sensor_mode=sensor_mode,
            timestamp=datetime.now(timezone.utc)
        )
