"""
Emergency Response and Email Notification Endpoints (Phase 5B).
"""

from fastapi import APIRouter
from app.schemas.emergency import EmergencyStatusResponse, EmailStatusResponse
from app.services.emergency_response_service import EmergencyResponseService
from app.services.email_notification_service import EmailNotificationService

router = APIRouter()


@router.get("/status", response_model=EmergencyStatusResponse, summary="Get Emergency Confirmation Status")
def get_emergency_status():
    """
    Get current Phase 5B emergency response status including confirmation state,
    temporal hits, visible occupants, and environmental corroboration context.
    """
    return EmergencyResponseService.get_status()


@router.get("/email/status", response_model=EmailStatusResponse, summary="Get Emergency Email Status")
def get_email_status():
    """
    Get safe status of the SMTP emergency email service without exposing credentials.
    """
    return EmailNotificationService.get_status()
