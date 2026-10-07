"""
Alerts management and acknowledgement API endpoints.
"""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.alert import Alert
from app.schemas.risk import AlertResponse

router = APIRouter()


@router.get("", response_model=List[AlertResponse])
def get_alerts(
    unacknowledged_only: bool = Query(default=False),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieve list of generated emergency and hazard alerts."""
    query = db.query(Alert).order_by(Alert.created_at.desc())
    if unacknowledged_only:
        query = query.filter(Alert.acknowledged == False)
    alerts = query.limit(limit).all()
    return [AlertResponse.model_validate(a) for a in alerts]


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an active alert and record the timestamp."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return AlertResponse.model_validate(alert)
