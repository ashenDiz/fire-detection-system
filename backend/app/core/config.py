"""
Centralized Configuration for Fire Detection and Emergency Alert System.
All risk thresholds, weights, and operational boundaries are configured here.
"""

from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve backend .env path relative to package root
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Intelligent Multi-Modal Fire Detection and Emergency Alert System"
    VERSION: str = "1.0.0-phase1"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./database/fire_system.db"
    
    # Input Range Constraints for Validation
    TEMP_MIN: float = 0.0
    TEMP_MAX: float = 120.0
    HUMIDITY_MIN: float = 0.0
    HUMIDITY_MAX: float = 100.0
    SMOKE_MIN: float = 0.0
    SMOKE_MAX: float = 100.0
    GAS_MIN: float = 0.0
    GAS_MAX: float = 100.0
    FLAME_MIN: float = 0.0
    FLAME_MAX: float = 100.0

    # Risk Fusion Component Weights (Sum = 1.0)
    WEIGHT_TEMPERATURE: float = 0.20
    WEIGHT_SMOKE: float = 0.25
    WEIGHT_GAS: float = 0.15
    WEIGHT_FLAME: float = 0.20
    WEIGHT_CAMERA_FIRE: float = 0.20

    # Normalization Thresholds
    TEMP_SAFE_MAX: float = 30.0
    TEMP_ELEVATED_MAX: float = 50.0
    TEMP_HIGH_MAX: float = 70.0
    TEMP_CRITICAL: float = 100.0

    SMOKE_BASELINE_MAX: float = 15.0
    SMOKE_ELEVATED_MAX: float = 40.0
    SMOKE_HEAVY_MAX: float = 70.0

    GAS_AMBIENT_MAX: float = 20.0
    GAS_ELEVATED_MAX: float = 50.0
    GAS_HAZARD_MAX: float = 80.0

    FLAME_NEGLIGIBLE_MAX: float = 20.0
    FLAME_MODERATE_MAX: float = 60.0

    # Risk Level Boundaries
    RISK_LEVEL_SAFE_MAX: float = 29.99
    RISK_LEVEL_CAUTION_MAX: float = 49.99
    RISK_LEVEL_WARNING_MAX: float = 69.99
    RISK_LEVEL_HIGH_MAX: float = 84.99
    RISK_LEVEL_CRITICAL_MIN: float = 85.0

    # Computer Vision Thresholds (Phase 4 & Phase 5B)
    FIRE_CONFIDENCE_THRESHOLD: float = 0.30  # Raw visual candidate / bounding box detection threshold
    FIRE_CONFIRMATION_THRESHOLD: float = 0.50  # Stronger evidence threshold used by Phase 5B emergency logic
    FIRE_CONFIRMATION_WINDOW_SECONDS: float = 2.0  # Rolling observation window
    FIRE_CONFIRMATION_MIN_HITS: int = 2  # Minimum hits of >= 0.50 required within window
    FIRE_CLEAR_MIN_FRESH_NEGATIVE_SAMPLES: int = 5  # Fresh no-fire observations required to clear emergency
    EMERGENCY_SENSOR_MAX_AGE_SECONDS: float = 30.0  # Max age for sensor corroboration telemetry

    # Email Notification Configuration (Phase 5B)
    EMAIL_NOTIFICATIONS_ENABLED: bool = False
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    SMTP_REQUIRE_AUTH: bool = True
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    EMERGENCY_EMAIL_FROM: str = ""
    EMERGENCY_EMAIL_TO: str = ""

    # Camera & Video Streaming Configuration (Phase 2)
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_TARGET_FPS: int = 30
    JPEG_QUALITY: int = 80

    # Person Detection Configuration (Phase 3)
    PERSON_MODEL_PATH: str = "yolo26n.pt"
    PERSON_CONFIDENCE_THRESHOLD: float = 0.50
    PERSON_INFERENCE_FPS: float = 5.0
    PERSON_IMAGE_SIZE: int = 640
    PERSON_DEVICE: str = "cpu"
    PERSON_STALE_TIMEOUT: float = 2.0

    # Fire & Smoke Detection Configuration (Phase 4)
    FIRE_MODEL_PATH: str = "models/best.pt"
    FIRE_INFERENCE_FPS: float = 3.0
    FIRE_IMAGE_SIZE: int = 640
    FIRE_DEVICE: str = "cpu"
    FIRE_STALE_TIMEOUT: float = 2.0



    # Predefined Academic Research Test Scenarios
    RESEARCH_SCENARIOS: Dict[str, Dict[str, Any]] = {
        "NORMAL": {
            "name": "Normal Ambient Condition",
            "description": "Typical baseline room temperature and humidity with no smoke, gas, or flame.",
            "temperature": 28.0,
            "humidity": 60.0,
            "smoke_level": 5.0,
            "gas_level": 4.0,
            "flame_level": 0.0,
        },
        "GAS_LEAK": {
            "name": "Gas Leak Hazard",
            "description": "High flammable or toxic gas concentration without combustion or optical flame.",
            "temperature": 30.0,
            "humidity": 60.0,
            "smoke_level": 15.0,
            "gas_level": 90.0,
            "flame_level": 0.0,
        },
        "POSSIBLE_FIRE": {
            "name": "Possible Developing Fire",
            "description": "Elevated temperature, moderate smoke obscuration, and intermittent flame presence.",
            "temperature": 55.0,
            "humidity": 40.0,
            "smoke_level": 65.0,
            "gas_level": 40.0,
            "flame_level": 60.0,
        },
        "CRITICAL_FIRE": {
            "name": "Critical Fire Emergency",
            "description": "High temperature, dense smoke concentration, severe gas, and continuous active flame.",
            "temperature": 75.0,
            "humidity": 25.0,
            "smoke_level": 95.0,
            "gas_level": 70.0,
            "flame_level": 95.0,
        },
    }

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
