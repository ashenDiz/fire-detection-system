"""
Test Legacy SQLite Database Migration (Pre-Hardening Schema).
Verifies:
1. Pre-hardening legacy schema (without people_detection_available column) can be upgraded.
2. people_detection_available columns are added via ALTER TABLE.
3. Existing legacy rows and stored people count values remain completely intact.
4. Application can insert an unavailable assessment using storage sentinel 0 + availability=False.
5. API response schemas (RiskAssessmentResponse, AlertResponse) expose people_detected=null when availability=False.
6. Database isolation: NEVER touches database/fire_system.db.
"""

import os
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.risk import RiskAssessment
from app.models.alert import Alert
from app.models.event import DetectionEvent
from app.schemas.risk import RiskAssessmentResponse, AlertResponse


def test_legacy_database_migration_isolated(tmp_path):
    # Ensure test runs against an isolated temporary SQLite database
    migration_db_path = os.path.abspath(str(tmp_path / "legacy_test.db"))
    dev_db_path = os.path.abspath(settings.DATABASE_URL.replace("sqlite:///", ""))
    assert migration_db_path != dev_db_path, "CRITICAL: Must not run against development database!"

    migration_url = f"sqlite:///{migration_db_path}"
    engine = create_engine(migration_url, connect_args={"check_same_thread": False})

    # Step 1: Create legacy pre-hardening schema (WITHOUT people_detection_available columns)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature FLOAT NOT NULL,
                humidity FLOAT NOT NULL,
                flame_level FLOAT NOT NULL,
                smoke_level FLOAT NOT NULL,
                gas_level FLOAT NOT NULL,
                input_source VARCHAR(50) NOT NULL,
                scenario_tag VARCHAR(50),
                created_at DATETIME NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_reading_id INTEGER,
                overall_risk_score FLOAT NOT NULL,
                risk_level VARCHAR(50) NOT NULL,
                temp_risk FLOAT NOT NULL,
                smoke_risk FLOAT NOT NULL,
                gas_risk FLOAT NOT NULL,
                flame_risk FLOAT NOT NULL,
                camera_fire_risk FLOAT,
                people_detected INTEGER NOT NULL DEFAULT 0,
                emergency_priority VARCHAR(50) NOT NULL,
                contributing_factors TEXT NOT NULL,
                created_at DATETIME NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hazard_signature VARCHAR(50),
                sensor_reading_id INTEGER,
                risk_assessment_id INTEGER,
                risk_score FLOAT NOT NULL,
                risk_level VARCHAR(50) NOT NULL,
                alert_type VARCHAR(50) NOT NULL,
                message VARCHAR(255) NOT NULL,
                people_detected INTEGER NOT NULL DEFAULT 0,
                fire_detected BOOLEAN NOT NULL DEFAULT 0,
                fire_confidence FLOAT,
                acknowledged BOOLEAN NOT NULL DEFAULT 0,
                acknowledged_at DATETIME,
                created_at DATETIME NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fire_detected BOOLEAN NOT NULL DEFAULT 0,
                fire_confidence FLOAT,
                people_count INTEGER NOT NULL DEFAULT 0,
                risk_score FLOAT NOT NULL,
                risk_level VARCHAR(50) NOT NULL,
                camera_status VARCHAR(50) NOT NULL,
                created_at DATETIME NOT NULL
            );
        """))

        # Step 2: Insert historical pre-hardening rows
        conn.execute(text("""
            INSERT INTO sensor_readings (temperature, humidity, flame_level, smoke_level, gas_level, input_source, created_at)
            VALUES (25.0, 50.0, 0.0, 5.0, 10.0, 'manual_simulation', '2026-10-01 12:00:00');
        """))
        # Historical row with people_detected = 3
        conn.execute(text("""
            INSERT INTO risk_assessments (
                sensor_reading_id, overall_risk_score, risk_level, temp_risk, smoke_risk, gas_risk, flame_risk,
                people_detected, emergency_priority, contributing_factors, created_at
            )
            VALUES (1, 80.0, 'HIGH RISK', 30.0, 40.0, 20.0, 10.0, 3, 'HIGH', '[]', '2026-10-01 12:00:00');
        """))
        conn.execute(text("""
            INSERT INTO alerts (
                sensor_reading_id, risk_assessment_id, risk_score, risk_level, alert_type, message,
                people_detected, fire_detected, acknowledged, created_at
            )
            VALUES (1, 1, 80.0, 'HIGH RISK', 'HIGH', 'High risk alert', 3, 0, 0, '2026-10-01 12:00:00');
        """))
        conn.execute(text("""
            INSERT INTO detection_events (
                fire_detected, fire_confidence, people_count, risk_score, risk_level, camera_status, created_at
            )
            VALUES (0, NULL, 0, 80.0, 'HIGH RISK', 'Connected', '2026-10-01 12:00:00');
        """))
        conn.commit()

    # Step 3: Run the database schema migration (exactly as performed at startup in main.py)
    with engine.connect() as conn:
        cols_alerts = [row[1] for row in conn.execute(text("PRAGMA table_info(alerts);")).fetchall()]
        if "people_detection_available" not in cols_alerts:
            conn.execute(text("ALTER TABLE alerts ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()
        if "fire_detection_available" not in cols_alerts:
            conn.execute(text("ALTER TABLE alerts ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()

        cols_risk = [row[1] for row in conn.execute(text("PRAGMA table_info(risk_assessments);")).fetchall()]
        if "people_detection_available" not in cols_risk:
            conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()
        if "fire_detection_available" not in cols_risk:
            conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()
        if "fire_confidence" not in cols_risk:
            conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN fire_confidence FLOAT;"))
            conn.commit()
        if "visual_smoke_detected" not in cols_risk:
            conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN visual_smoke_detected BOOLEAN;"))
            conn.commit()
        if "visual_smoke_confidence" not in cols_risk:
            conn.execute(text("ALTER TABLE risk_assessments ADD COLUMN visual_smoke_confidence FLOAT;"))
            conn.commit()

        cols_events = [row[1] for row in conn.execute(text("PRAGMA table_info(detection_events);")).fetchall()]
        if "people_detection_available" not in cols_events:
            conn.execute(text("ALTER TABLE detection_events ADD COLUMN people_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()
        if "fire_detection_available" not in cols_events:
            conn.execute(text("ALTER TABLE detection_events ADD COLUMN fire_detection_available BOOLEAN NOT NULL DEFAULT 0;"))
            conn.commit()

    # Step 4: Verify columns were added and historical rows preserved
    with engine.connect() as conn:
        cols_risk_after = [row[1] for row in conn.execute(text("PRAGMA table_info(risk_assessments);")).fetchall()]
        assert "people_detection_available" in cols_risk_after
        assert "fire_detection_available" in cols_risk_after
        assert "fire_confidence" in cols_risk_after

        legacy_row = conn.execute(text("SELECT people_detected, people_detection_available, fire_detection_available, fire_confidence FROM risk_assessments WHERE id = 1;")).fetchone()
        assert legacy_row is not None
        assert legacy_row[0] == 3, "Historical people_detected count must be preserved as 3!"
        assert legacy_row[1] == 0, "Migrated row should have default people_detection_available = 0 (False)"
        assert legacy_row[2] == 0, "Migrated row should have default fire_detection_available = 0 (False)"
        assert legacy_row[3] is None, "Migrated row should have default fire_confidence = NULL"

        cols_events_after = [row[1] for row in conn.execute(text("PRAGMA table_info(detection_events);")).fetchall()]
        assert "fire_detection_available" in cols_events_after

        legacy_event_row = conn.execute(text("SELECT fire_detection_available, fire_detected, fire_confidence FROM detection_events WHERE id = 1;")).fetchone()
        assert legacy_event_row is not None
        assert legacy_event_row[0] == 0, "Migrated detection event must have default fire_detection_available = 0 (False)"
        assert legacy_event_row[1] == 0, "DB storage sentinel retained as 0 (False)"
        assert legacy_event_row[2] is None, "Migrated event fire_confidence is NULL"

    # Step 5: Test application ORM interaction against migrated database
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 5a. Read historical assessment through ORM and serialize to Pydantic
        orm_legacy = session.query(RiskAssessment).filter(RiskAssessment.id == 1).first()
        assert orm_legacy is not None
        assert orm_legacy.people_detected == 3
        assert orm_legacy.people_detection_available is False
        assert orm_legacy.fire_detection_available is False
        assert orm_legacy.fire_confidence is None
        assert orm_legacy.camera_fire_risk is None

        # When serialized to API schema, since people_detection_available is False,
        # @model_validator converts people_detected to None (never falsely confirmed!)
        schema_legacy = RiskAssessmentResponse.model_validate(orm_legacy)
        assert schema_legacy.people_detection_available is False
        assert schema_legacy.people_detected is None
        assert schema_legacy.fire_detection_available is False
        assert schema_legacy.camera_fire_confidence is None
        assert schema_legacy.camera_fire_risk is None

        # 5b. Verify historical Alert DB storage sentinel is normalized through AlertResponse
        orm_alert = session.query(Alert).filter(Alert.id == 1).first()
        assert orm_alert is not None
        assert orm_alert.fire_detection_available is False
        assert orm_alert.fire_detected is False  # Sentinel in DB
        assert orm_alert.fire_confidence is None

        schema_alert = AlertResponse.model_validate(orm_alert)
        assert schema_alert.fire_detection_available is False
        assert schema_alert.fire_detected is None  # Normalized to None when AI unavailable
        assert schema_alert.fire_confidence is None

        # 5c. Verify historical DetectionEvent DB storage sentinel is normalized through DetectionEventResponse
        from app.schemas.risk import DetectionEventResponse
        orm_event = session.query(DetectionEvent).filter(DetectionEvent.id == 1).first()
        assert orm_event is not None
        assert orm_event.fire_detection_available is False
        assert orm_event.fire_detected is False  # Sentinel in DB
        assert orm_event.fire_confidence is None

        schema_event = DetectionEventResponse.model_validate(orm_event)
        assert schema_event.fire_detection_available is False
        assert schema_event.fire_detected is None  # Normalized to None when AI unavailable
        assert schema_event.fire_confidence is None

        # 5d. Insert an unavailable assessment using storage sentinel 0 + availability=False
        new_assessment = RiskAssessment(
            sensor_reading_id=1,
            overall_risk_score=90.0,
            risk_level="CRITICAL",
            temp_risk=80.0,
            smoke_risk=85.0,
            gas_risk=20.0,
            flame_risk=75.0,
            camera_fire_risk=None,
            fire_detection_available=False,
            fire_confidence=None,
            visual_smoke_detected=None,
            visual_smoke_confidence=None,
            people_detected=0,  # Storage sentinel (NON-NULL)
            people_detection_available=False,  # Explicitly unavailable
            emergency_priority="CRITICAL",
            contributing_factors="[]",
            created_at=datetime.now(timezone.utc)
        )
        session.add(new_assessment)
        session.commit()
        session.refresh(new_assessment)

        assert new_assessment.id is not None
        assert new_assessment.people_detected == 0
        assert new_assessment.people_detection_available is False
        assert new_assessment.fire_detection_available is False
        assert new_assessment.fire_confidence is None

        # Verify API response exposes people_detected = null, camera_fire_confidence = null
        schema_new = RiskAssessmentResponse.model_validate(new_assessment)
        assert schema_new.people_detection_available is False
        assert schema_new.people_detected is None
        assert schema_new.fire_detection_available is False
        assert schema_new.camera_fire_risk is None
        assert schema_new.camera_fire_confidence is None

    finally:
        session.close()
        engine.dispose()
