"""
Main FastAPI Application Entrypoint.
Intelligent Multi-Modal Fire Detection and Emergency Alert System.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database.session import Base, engine
from app.api.v1 import api_v1_router



from sqlalchemy import text

@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys, os
    if "pytest" not in sys.modules and not os.environ.get("PYTEST_CURRENT_TEST"):
        # Startup: Ensure database tables exist without fabricating sensor telemetry
        Base.metadata.create_all(bind=engine)

        # Ensure schema synchronization for SQLite development database
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(alerts);")).fetchall()]
            if cols and "hazard_signature" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN hazard_signature VARCHAR(50);"))
                conn.commit()
            if cols and "sensor_reading_id" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN sensor_reading_id INTEGER REFERENCES sensor_readings(id);"))
                conn.commit()
            if cols and "risk_assessment_id" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN risk_assessment_id INTEGER REFERENCES risk_assessments(id);"))
                conn.commit()
            if cols and "people_detection_available" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()
            if cols and "fire_detection_available" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()

            cols_risk = [row[1] for row in conn.execute(text("PRAGMA table_info(risk_assessments);")).fetchall()]
            if cols_risk and "people_detection_available" not in cols_risk:
                conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()
            if cols_risk and "fire_detection_available" not in cols_risk:
                conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()
            if cols_risk and "fire_confidence" not in cols_risk:
                conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN fire_confidence FLOAT;"))
                conn.commit()
            if cols_risk and "visual_smoke_detected" not in cols_risk:
                conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN visual_smoke_detected BOOLEAN;"))
                conn.commit()
            if cols_risk and "visual_smoke_confidence" not in cols_risk:
                conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN visual_smoke_confidence FLOAT;"))
                conn.commit()

            cols_events = [row[1] for row in conn.execute(text("PRAGMA table_info(detection_events);")).fetchall()]
            if cols_events and "people_detection_available" not in cols_events:
                conn.execute(text("ALTER TABLE detection_events ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()
            if cols_events and "fire_detection_available" not in cols_events:
                conn.execute(text("ALTER TABLE detection_events ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
                conn.commit()

        # Phase 3: Safely attempt YOLO person model load once at startup in production runtime
        try:
            from app.services.person_detection_service import PersonDetectionService
            PersonDetectionService.initialize_model()
        except Exception:
            pass

        # Phase 4: Safely attempt custom D-Fire YOLO model load once at startup in production runtime
        try:
            from app.services.fire_detection_service import FireDetectionService
            FireDetectionService.initialize_model()
        except Exception:
            pass

        # Phase 5B: Safely start background emergency monitor worker in production runtime
        try:
            from app.services.emergency_response_service import EmergencyResponseService
            EmergencyResponseService.start_monitor()
        except Exception:
            pass

    yield

    # Clean resource release on application shutdown
    try:
        from app.services.emergency_response_service import EmergencyResponseService
        EmergencyResponseService.stop_monitor()
    except Exception:
        pass

    try:
        from app.services.person_detection_service import PersonDetectionService
        PersonDetectionService.stop_worker()
    except Exception:
        pass

    try:
        from app.services.fire_detection_service import FireDetectionService
        FireDetectionService.stop_worker()
    except Exception:
        pass

    from app.services.camera_service import CameraService
    CameraService.stop_camera()



app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Academic research prototype combining environmental sensor fusion and computer vision.",
    lifespan=lifespan
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs"
    }
