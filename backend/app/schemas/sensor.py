"""
Pydantic schemas for sensor inputs, validation, and responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from app.core.config import settings


class SensorReadingBase(BaseModel):
    temperature: float = Field(
        ...,
        description="Temperature in degrees Celsius",
        ge=settings.TEMP_MIN,
        le=settings.TEMP_MAX
    )
    humidity: float = Field(
        ...,
        description="Relative humidity percentage",
        ge=settings.HUMIDITY_MIN,
        le=settings.HUMIDITY_MAX
    )
    flame_level: float = Field(
        ...,
        description="Flame presence indicator (0-100%)",
        ge=settings.FLAME_MIN,
        le=settings.FLAME_MAX
    )
    smoke_level: float = Field(
        ...,
        description="Smoke obscuration level (0-100%)",
        ge=settings.SMOKE_MIN,
        le=settings.SMOKE_MAX
    )
    gas_level: float = Field(
        ...,
        description="Dangerous gas concentration (0-100%)",
        ge=settings.GAS_MIN,
        le=settings.GAS_MAX
    )


class SensorReadingCreate(SensorReadingBase):
    input_source: str = Field(default="manual_simulation", description="Source of input data")
    scenario_tag: Optional[str] = Field(default=None, description="Identifier if from a research scenario")

    @field_validator("input_source")
    def validate_source(cls, v: str) -> str:
        valid_sources = ["manual_simulation", "esp32", "arduino", "hardware"]
        if v not in valid_sources:
            raise ValueError(f"Input source must be one of: {valid_sources}")
        return v


class SensorReadingResponse(SensorReadingBase):
    id: int
    input_source: str
    scenario_tag: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ScenarioPreset(BaseModel):
    id: str
    name: str
    description: str
    temperature: float
    humidity: float
    smoke_level: float
    gas_level: float
    flame_level: float
