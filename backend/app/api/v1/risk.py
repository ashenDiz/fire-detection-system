"""
Risk calculation and current assessment endpoints.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.risk import RiskAssessment
from app.schemas.risk import RiskAssessmentResponse

router = APIRouter()


@router.get("/current", response_model=RiskAssessmentResponse)
def get_current_risk(db: Session = Depends(get_db)):
    """Retrieve the most recent multi-source risk assessment calculation."""
    assessment = db.query(RiskAssessment).order_by(RiskAssessment.created_at.desc()).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="No risk assessment has been calculated yet.")

    factors = json.loads(assessment.contributing_factors) if isinstance(assessment.contributing_factors, str) else assessment.contributing_factors

    return RiskAssessmentResponse(
        id=assessment.id,
        sensor_reading_id=assessment.sensor_reading_id,
        overall_risk_score=assessment.overall_risk_score,
        risk_level=assessment.risk_level,
        temp_risk=assessment.temp_risk,
        smoke_risk=assessment.smoke_risk,
        gas_risk=assessment.gas_risk,
        flame_risk=assessment.flame_risk,
        camera_fire_risk=assessment.camera_fire_risk,
        fire_detection_available=getattr(assessment, "fire_detection_available", False),
        camera_fire_confidence=getattr(assessment, "fire_confidence", None),
        fire_confidence=getattr(assessment, "fire_confidence", None),
        visual_smoke_detected=getattr(assessment, "visual_smoke_detected", None),
        visual_smoke_confidence=getattr(assessment, "visual_smoke_confidence", None),
        people_detected=assessment.people_detected,
        people_detection_available=getattr(assessment, "people_detection_available", False),
        emergency_priority=assessment.emergency_priority,
        contributing_factors=factors,
        created_at=assessment.created_at
    )
