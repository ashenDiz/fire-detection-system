"""
System diagnostics and transparency endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.schemas.system import SystemStatusResponse
from app.services.system_service import SystemService

router = APIRouter()


@router.get("/status", response_model=SystemStatusResponse)
def get_system_status(db: Session = Depends(get_db)):
    """Fetch health and availability status of all system components and AI models."""
    return SystemService.get_status(db)
