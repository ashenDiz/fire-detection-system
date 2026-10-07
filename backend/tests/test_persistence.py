"""
Persistence verification test across engine/backend restart.
Uses an isolated test database to verify restart persistence without touching dev database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.database.session import Base
from app.models.sensor import SensorReading
from app.models.risk import RiskAssessment
from app.services.risk_engine import RiskAssessmentService


def test_sqlite_persistence_across_restart():
    # Dedicated persistence test database
    db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_restart_persistence.db"))
    test_db_url = f"sqlite:///{db_file}"

    # Safety assertion
    assert test_db_url != settings.DATABASE_URL
    if os.path.exists(db_file):
        os.remove(db_file)

    p_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=p_engine)
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=p_engine)

    # 1. Write record in Session A
    db1 = SessionMaker()
    reading = SensorReading(
        temperature=42.5,
        humidity=65.0,
        flame_level=10.0,
        smoke_level=20.0,
        gas_level=15.0,
        input_source="manual_simulation",
        scenario_tag="PERSISTENCE_TEST_AUDIT"
    )
    db1.add(reading)
    db1.commit()
    db1.refresh(reading)
    saved_id = reading.id

    assessment, alert = RiskAssessmentService.evaluate_and_record(reading, db1)
    saved_risk_score = assessment.overall_risk_score
    db1.close()

    # 2. Simulate backend shutdown & restart: dispose engine connection pool
    p_engine.dispose()

    # 3. Open completely new Session B from a new engine instance
    new_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    NewSessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=new_engine)
    db2 = NewSessionMaker()

    fetched_reading = db2.query(SensorReading).filter(SensorReading.id == saved_id).first()
    assert fetched_reading is not None, "Reading must persist after engine restart"
    assert fetched_reading.temperature == 42.5
    assert fetched_reading.scenario_tag == "PERSISTENCE_TEST_AUDIT"

    fetched_assessment = db2.query(RiskAssessment).filter(RiskAssessment.sensor_reading_id == saved_id).first()
    assert fetched_assessment is not None, "Risk assessment must persist after restart"
    assert fetched_assessment.overall_risk_score == saved_risk_score
    assert fetched_assessment.sensor_reading_id == saved_id

    db2.close()
    new_engine.dispose()

    # Cleanup
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass


if __name__ == "__main__":
    test_sqlite_persistence_across_restart()
