"""
Sensor reading submission, history, and current readout endpoints.
"""

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.sensor import SensorReading
from app.schemas.sensor import SensorReadingCreate, SensorReadingResponse
from app.schemas.risk import SensorSubmissionResult, RiskAssessmentResponse, AlertResponse
from app.services.sensor_provider import get_sensor_provider
from app.services.risk_engine import RiskAssessmentService

router = APIRouter()


@router.post("/readings", response_model=SensorSubmissionResult, status_code=status.HTTP_201_CREATED)
def submit_sensor_reading(
    reading_in: SensorReadingCreate,
    db: Session = Depends(get_db)
):
    """
    Submit manual or telemetry sensor data.
    Validates data, persists to database, executes risk fusion engine,
    and generates alerts if necessary.
    """
    provider = get_sensor_provider()
    db_reading = provider.record_reading(reading_in, db)

    # Phase 5A Coherent Camera Session Snapshot Guard
    from app.services.camera_service import CameraService
    from app.services.person_detection_service import PersonDetectionService
    from app.services.fire_detection_service import FireDetectionService

    generation_before = CameraService.get_generation()
    fresh_people = PersonDetectionService.get_fresh_person_count()
    fire_summary = FireDetectionService.get_fresh_summary()
    generation_after = CameraService.get_generation()

    if generation_before != generation_after:
        # Camera session changed during snapshot acquisition; treat modalities as unavailable
        fresh_people = None
        camera_fire_conf = None
        visual_smoke_detected = None
        visual_smoke_conf = None
    else:
        camera_fire_conf = (
            fire_summary["fire_confidence"]
            if fire_summary.get("detection_available")
            else None
        )
        visual_smoke_detected = (
            fire_summary["smoke_detected"]
            if fire_summary.get("detection_available")
            else None
        )
        visual_smoke_conf = (
            fire_summary["smoke_confidence"]
            if fire_summary.get("detection_available")
            else None
        )

    # Evaluate risk using RiskAssessmentService
    db_assessment, db_alert = RiskAssessmentService.evaluate_and_record(
        reading=db_reading,
        db=db,
        camera_fire_conf=camera_fire_conf,
        people_count=fresh_people,
        visual_smoke_detected=visual_smoke_detected,
        visual_smoke_conf=visual_smoke_conf
    )

    factors = json.loads(db_assessment.contributing_factors) if isinstance(db_assessment.contributing_factors, str) else db_assessment.contributing_factors

    risk_response = RiskAssessmentResponse(
        id=db_assessment.id,
        sensor_reading_id=db_assessment.sensor_reading_id,
        overall_risk_score=db_assessment.overall_risk_score,
        risk_level=db_assessment.risk_level,
        temp_risk=db_assessment.temp_risk,
        smoke_risk=db_assessment.smoke_risk,
        gas_risk=db_assessment.gas_risk,
        flame_risk=db_assessment.flame_risk,
        camera_fire_risk=db_assessment.camera_fire_risk,
        fire_detection_available=getattr(db_assessment, "fire_detection_available", False),
        camera_fire_confidence=getattr(db_assessment, "fire_confidence", None),
        fire_confidence=getattr(db_assessment, "fire_confidence", None),
        visual_smoke_detected=getattr(db_assessment, "visual_smoke_detected", None),
        visual_smoke_confidence=getattr(db_assessment, "visual_smoke_confidence", None),
        people_detected=db_assessment.people_detected,
        people_detection_available=getattr(db_assessment, "people_detection_available", False),
        emergency_priority=db_assessment.emergency_priority,
        contributing_factors=factors,
        created_at=db_assessment.created_at
    )

    alert_response = None
    if db_alert is not None:
        alert_response = AlertResponse.model_validate(db_alert)

    return SensorSubmissionResult(
        reading=SensorReadingResponse.model_validate(db_reading),
        risk=risk_response,
        active_alert=alert_response
    )


@router.get("/latest", response_model=SensorReadingResponse)
def get_latest_sensor_reading(db: Session = Depends(get_db)):
    """Retrieve the most recent environmental sensor reading."""
    reading = db.query(SensorReading).order_by(SensorReading.created_at.desc()).first()
    if not reading:
        raise HTTPException(status_code=404, detail="No sensor readings available.")
    return SensorReadingResponse.model_validate(reading)


@router.get("/history", response_model=List[SensorReadingResponse])
def get_sensor_history(
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieve recent historical sensor readings."""
    readings = db.query(SensorReading).order_by(SensorReading.created_at.desc()).limit(limit).all()
    return [SensorReadingResponse.model_validate(r) for r in readings]
