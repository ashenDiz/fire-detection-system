"""
Pydantic schemas for Phase 3 Person Detection and Tracking.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class PersonDetectionItem(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixel coordinates")
    confidence: float = Field(..., description="Detection confidence score (0.0 - 1.0)")
    class_name: str = Field(default="person", description="Detected class label")


class PersonStatusResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether YOLO person detection model is loaded and ready")
    status: str = Field(..., description="Operational status: Loaded, Not Loaded, or Error: <msg>")
    model_name: str = Field(..., description="Name or path of the loaded detection model")
    enabled: bool = Field(..., description="Whether background inference worker is currently running")
    confidence_threshold: float = Field(..., description="Configured confidence threshold for person filtering")
    people_count: Optional[int] = Field(default=None, description="Number of people currently visible in latest valid frame, or null if unavailable")
    detection_available: bool = Field(default=False, description="Whether person detection is actively available and fresh")
    last_inference_at: Optional[datetime] = Field(default=None, description="Timestamp of latest inference run")
    inference_fps: float = Field(default=0.0, description="Measured inference throughput in FPS")
    inference_latency_ms: float = Field(default=0.0, description="Measured single-frame inference latency in milliseconds")
    device: str = Field(default="cpu", description="Compute device used for inference (cpu / cuda)")


class PersonDetectionLatestResponse(BaseModel):
    model_status: str = Field(..., description="Status of person AI model")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    camera_connected: bool = Field(..., description="Whether camera is actively connected")
    people_count: Optional[int] = Field(default=None, description="Number of people currently detected in latest frame, or null if unavailable")
    detection_available: bool = Field(default=False, description="Whether person detection is actively available and fresh")
    detections: List[PersonDetectionItem] = Field(default_factory=list, description="List of bounding boxes and confidences")
    last_inference_at: Optional[datetime] = Field(default=None, description="Timestamp of latest inference")
    inference_latency_ms: float = Field(default=0.0, description="Latency in milliseconds")
    stale: bool = Field(default=False, description="Whether detections are stale or unavailable")


class FireDetectionItem(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixel coordinates")
    confidence: float = Field(..., description="Detection confidence score (0.0 - 1.0)")
    class_name: str = Field(..., description="Detected class label: 'fire' or 'smoke'")


class FireStatusResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether YOLO fire/smoke detection model is loaded and ready")
    status: str = Field(..., description="Operational status: Loaded, Not Loaded, or Error: <msg>")
    model_name: str = Field(..., description="Name or path of the loaded detection model")
    enabled: bool = Field(..., description="Whether background inference worker is currently running")
    confidence_threshold: float = Field(..., description="Configured confidence threshold for fire/smoke filtering")
    detection_available: bool = Field(default=False, description="Whether fire/smoke detection is actively available and fresh")
    fire_detected: Optional[bool] = Field(default=None, description="True if fire detected, False if confirmed no fire, null if unavailable")
    smoke_detected: Optional[bool] = Field(default=None, description="True if smoke detected, False if confirmed no smoke, null if unavailable")
    fire_confidence: Optional[float] = Field(default=None, description="Max confidence score among detected fire boxes, 0.0 if confirmed absent, null if unavailable")
    smoke_confidence: Optional[float] = Field(default=None, description="Max confidence score among detected smoke boxes, 0.0 if confirmed absent, null if unavailable")
    last_inference_at: Optional[datetime] = Field(default=None, description="Timestamp of latest inference run")
    inference_fps: float = Field(default=0.0, description="Measured inference throughput in FPS")
    inference_latency_ms: float = Field(default=0.0, description="Measured single-frame inference latency in milliseconds")
    device: str = Field(default="cpu", description="Compute device used for inference (cpu / cuda)")


class FireDetectionLatestResponse(BaseModel):
    model_status: str = Field(..., description="Status of fire AI model")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    camera_connected: bool = Field(..., description="Whether camera is actively connected")
    detection_available: bool = Field(default=False, description="Whether fire detection is actively available and fresh")
    fire_detected: Optional[bool] = Field(default=None, description="True if fire detected, False if confirmed no fire, null if unavailable")
    smoke_detected: Optional[bool] = Field(default=None, description="True if smoke detected, False if confirmed no smoke, null if unavailable")
    fire_confidence: Optional[float] = Field(default=None, description="Max confidence of detected fire, null if unavailable")
    smoke_confidence: Optional[float] = Field(default=None, description="Max confidence of detected smoke, null if unavailable")
    detections: List[FireDetectionItem] = Field(default_factory=list, description="List of fire/smoke bounding boxes and confidences")
    last_inference_at: Optional[datetime] = Field(default=None, description="Timestamp of latest inference")
    inference_latency_ms: float = Field(default=0.0, description="Latency in milliseconds")
    stale: bool = Field(default=False, description="Whether detections are stale or unavailable")

