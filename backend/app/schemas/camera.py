"""
Pydantic schemas for Phase 2 Camera control and streaming.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CameraStartRequest(BaseModel):
    camera_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional hardware index for webcam (must be >= 0, defaults to CAMERA_INDEX from settings)"
    )



class CameraStatusResponse(BaseModel):
    connected: bool = Field(..., description="Whether webcam hardware is actively open and capturing")
    status: str = Field(..., description="Operational status: Connected, Disconnected, or Error: <msg>")
    camera_index: int = Field(..., description="Hardware device index")
    width: int = Field(..., description="Stream width resolution")
    height: int = Field(..., description="Stream height resolution")
    fps: float = Field(..., description="Stream target or measured frames per second")
    error: Optional[str] = Field(default=None, description="Diagnostic error message if camera failed")
