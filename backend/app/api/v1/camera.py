"""
Camera API Router for Phase 2: Webcam start/stop, status reporting, and MJPEG video streaming.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.camera import CameraStartRequest, CameraStatusResponse
from app.services.camera_service import CameraService

router = APIRouter()


@router.post(
    "/start",
    response_model=CameraStatusResponse,
    summary="Start webcam capture",
    description="Initializes OpenCV VideoCapture on physical webcam index and starts frame capture worker."
)
def start_camera(payload: Optional[CameraStartRequest] = None):
    camera_index = payload.camera_index if payload else None
    if camera_index is not None and camera_index < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Camera index must be a non-negative integer (>= 0)."
        )
    info = CameraService.start_camera(camera_index)


    if not info["connected"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=info.get("error", "Unable to open webcam.")
        )

    return CameraStatusResponse(**info)


@router.post(
    "/stop",
    response_model=CameraStatusResponse,
    summary="Stop webcam capture",
    description="Stops capture worker and releases OpenCV VideoCapture hardware resource."
)
def stop_camera():
    info = CameraService.stop_camera()
    return CameraStatusResponse(**info)


@router.get(
    "/status",
    response_model=CameraStatusResponse,
    summary="Get camera status",
    description="Returns current webcam connection state, resolution, frame rate, and diagnostics."
)
def get_camera_status():
    info = CameraService.get_camera_info()
    return CameraStatusResponse(**info)


@router.get(
    "/stream",
    summary="Live MJPEG video stream",
    description="Streams multipart/x-mixed-replace JPEG frames directly from OpenCV webcam."
)
def get_camera_stream():
    if not CameraService.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webcam is disconnected. Please start the camera to enable video streaming."
        )

    return StreamingResponse(
        CameraService.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
