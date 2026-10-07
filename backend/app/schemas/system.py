"""
Pydantic schemas for system diagnostics, health checks, and transparency reports.
"""

from datetime import datetime
from pydantic import BaseModel


class SystemStatusResponse(BaseModel):
    backend: str = "Online"
    database: str = "Connected"
    camera: str = "Disconnected"
    person_model: str = "Not Loaded"
    fire_model: str = "Fire AI Model Not Loaded"
    sensor_mode: str = "Manual Simulation"
    timestamp: datetime
