"""
API router for Computer Vision Detection modules.
Phase 3: Real YOLO Person Detection & Visible Occupancy Counting.
Phase 4: Real Custom D-Fire YOLO Fire & Smoke Detection.
"""

from fastapi import APIRouter
from app.schemas.detection import (
    PersonStatusResponse,
    PersonDetectionLatestResponse,
    FireStatusResponse,
    FireDetectionLatestResponse,
)
from app.services.person_detection_service import PersonDetectionService
from app.services.fire_detection_service import FireDetectionService

router = APIRouter()


@router.get("/person/status", response_model=PersonStatusResponse)
def get_person_detection_status():
    """
    Return operational and diagnostic status of the YOLO person detector.
    Reports model readiness, inference FPS, latency, and visible person count.
    """
    return PersonDetectionService.get_status()


@router.get("/person/latest", response_model=PersonDetectionLatestResponse)
def get_person_detection_latest():
    """
    Return latest bounding boxes and visible person count from recent camera frames.
    Does not fabricate detections. Transparently reports if detections are stale or camera is off.
    """
    return PersonDetectionService.get_latest()


@router.get("/fire/status", response_model=FireStatusResponse)
def get_fire_detection_status():
    """
    Return operational and diagnostic status of the custom D-Fire YOLO detector.
    Reports model readiness, inference FPS, latency, and fire/smoke detection availability.
    """
    return FireDetectionService.get_status()


@router.get("/fire/latest", response_model=FireDetectionLatestResponse)
def get_fire_detection_latest():
    """
    Return latest fire and smoke bounding boxes and detection summaries from recent camera frames.
    Does not fabricate detections. Transparently reports if detections are stale or camera is off.
    """
    return FireDetectionService.get_latest()

