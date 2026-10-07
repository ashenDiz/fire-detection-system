"""
Detection events history API endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.event import DetectionEvent
from app.schemas.risk import DetectionEventResponse

router = APIRouter()


@router.get("", response_model=List[DetectionEventResponse])
def get_detection_events(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieve state-change and incident detection events."""
    events = db.query(DetectionEvent).order_by(DetectionEvent.created_at.desc()).limit(limit).all()
    return [DetectionEventResponse.model_validate(e) for e in events]
