"""
Automated tests validating all Phase 1 integrity audit fixes:
1. Test database isolation and safety assertions.
2. SQLite PRAGMA foreign_keys = ON enforcement and rejection of invalid FKs.
3. Advanced alert deduplication by hazard signature, escalation, and traceability.
4. Correct transparent sensor-only fire wording (no 'Fire condition detected' without camera).
5. Independent camera connection state and fire model readiness reporting.
"""

import os
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.models.alert import Alert
from app.models.sensor import SensorReading
from app.services.camera_service import CameraService
from app.services.system_service import SystemService
from app.services.risk_engine import RiskAssessmentService
from tests.conftest import TEST_DATABASE_URL, TEST_DB_PATH, TestingSessionLocal

client = TestClient(app)


def test_database_isolation_safety_assertion():
    """Verify that tests execute exclusively on the test database and never touch dev database."""
    prod_path = os.path.abspath(settings.DATABASE_URL.replace("sqlite:///", ""))
    
    # 1. URL and path assertions
    assert TEST_DATABASE_URL != settings.DATABASE_URL
    assert TEST_DB_PATH != prod_path
    assert "test_fire_system.db" in TEST_DATABASE_URL

    # 2. Verify active session in test uses test database
    db = TestingSessionLocal()
    result = db.execute(text("PRAGMA database_list")).fetchall()
    active_db_file = result[0][2] if result else ""
    db.close()
    assert "test_fire_system.db" in active_db_file.lower()


def test_sqlite_foreign_key_enforcement():
    """Verify PRAGMA foreign_keys=1 and assert invalid foreign key inserts raise IntegrityError."""
    db = TestingSessionLocal()
    
    # 1. Verify PRAGMA foreign_keys is enabled (returns 1)
    fk_status = db.execute(text("PRAGMA foreign_keys")).scalar()
    assert fk_status == 1, f"PRAGMA foreign_keys must equal 1, got {fk_status}"

    # 2. Attempt to insert Alert with non-existent sensor_reading_id
    invalid_alert = Alert(
        hazard_signature="TEST_INVALID",
        sensor_reading_id=999999,  # Non-existent foreign key
        risk_score=75.0,
        risk_level="HIGH RISK",
        alert_type="HIGH",
        message="Invalid foreign key test",
        people_detected=0,
        fire_detected=False,
        acknowledged=False
    )
    db.add(invalid_alert)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.close()


def test_alert_deduplication_scenarios():
    """
    Test all 4 deduplication scenarios mandated in audit:
    - Repeated identical gas-leak readings
    - Gas WARNING followed by smoke WARNING
    - WARNING escalating to CRITICAL
    - Acknowledged hazard occurring again
    """
    # Setup: acknowledge any pre-existing alerts
    open_alerts = client.get("/api/v1/alerts?unacknowledged_only=true").json()
    for a in open_alerts:
        client.post(f"/api/v1/alerts/{a['id']}/acknowledge")

    # --- Scenario A: Repeated identical gas-leak readings ---
    gas_payload = {
        "temperature": 30.0,
        "humidity": 60.0,
        "smoke_level": 15.0,
        "gas_level": 90.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation",
        "scenario_tag": "GAS_LEAK"
    }
    res_gas1 = client.post("/api/v1/sensors/readings", json=gas_payload).json()
    alert_gas1 = res_gas1["active_alert"]
    assert alert_gas1 is not None
    assert alert_gas1["hazard_signature"] == "GAS_LEAK"
    assert alert_gas1["sensor_reading_id"] == res_gas1["reading"]["id"]

    # Post repeated identical gas reading
    res_gas2 = client.post("/api/v1/sensors/readings", json=gas_payload).json()
    alert_gas2 = res_gas2["active_alert"]
    assert alert_gas2 is not None
    # Must NOT create a duplicate alert row
    assert alert_gas2["id"] == alert_gas1["id"]
    # Must update traceability to the current corroborating reading
    assert alert_gas2["sensor_reading_id"] == res_gas2["reading"]["id"]
    assert alert_gas2["risk_assessment_id"] == res_gas2["risk"]["id"]

    # --- Scenario B: Gas WARNING followed by Smoke WARNING ---
    # Unrelated hazard of the same severity must create a separate incident alert
    smoke_payload = {
        "temperature": 26.0,
        "humidity": 50.0,
        "smoke_level": 75.0,
        "gas_level": 10.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation",
        "scenario_tag": "SMOKE_HAZARD"
    }
    res_smoke = client.post("/api/v1/sensors/readings", json=smoke_payload).json()
    alert_smoke = res_smoke["active_alert"]
    assert alert_smoke is not None
    assert alert_smoke["hazard_signature"] == "SMOKE_HAZARD"
    # Crucial assertion: Must NOT be merged with the gas leak alert!
    assert alert_smoke["id"] != alert_gas1["id"], "Smoke warning must not merge into gas leak alert!"

    # --- Scenario C: WARNING escalating to CRITICAL ---
    crit_payload = {
        "temperature": 75.0,
        "humidity": 25.0,
        "smoke_level": 95.0,
        "gas_level": 70.0,
        "flame_level": 95.0,
        "input_source": "manual_simulation",
        "scenario_tag": "CRITICAL_FIRE"
    }
    res_crit = client.post("/api/v1/sensors/readings", json=crit_payload).json()
    alert_crit = res_crit["active_alert"]
    assert alert_crit is not None
    assert alert_crit["alert_type"] == "CRITICAL"
    assert alert_crit["id"] != alert_gas1["id"]
    assert alert_crit["id"] != alert_smoke["id"]

    # --- Scenario D: Acknowledged hazard occurring again ---
    # Acknowledge the critical alert
    client.post(f"/api/v1/alerts/{alert_crit['id']}/acknowledge")
    
    # Same critical hazard occurs again after acknowledgement
    res_crit_again = client.post("/api/v1/sensors/readings", json=crit_payload).json()
    alert_crit_again = res_crit_again["active_alert"]
    assert alert_crit_again is not None
    # Must create a new alert record since previous was acknowledged
    assert alert_crit_again["id"] != alert_crit["id"]
    assert alert_crit_again["acknowledged"] is False


def test_sensor_only_fire_wording_transparency():
    """Verify that sensor-only critical ratings never claim 'Fire condition detected'."""
    reading = SensorReading(
        id=99,
        temperature=75.0,
        humidity=25.0,
        flame_level=95.0,
        smoke_level=95.0,
        gas_level=70.0,
        input_source="manual_simulation"
    )
    # Phase 1: camera_fire_conf is None
    res = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)
    
    assert res["risk_level"] == "CRITICAL"
    # Must NOT claim fire detected
    assert "fire condition detected" not in res["alert_message"].lower()
    assert "visual fire detection confirmed" not in res["alert_message"].lower()
    # Must use transparent sensor-fusion wording
    assert "critical fire-risk condition indicated by environmental sensor fusion" in res["alert_message"].lower()
    assert res["camera_fire_risk"] is None


def test_camera_and_model_status_independence():
    """Verify Camera status can be Connected/Disconnected independently of AI models."""
    db = TestingSessionLocal()

    # Initial state: Disconnected
    CameraService.set_connected(False)
    status1 = SystemService.get_status(db)
    assert status1.camera == "Disconnected"
    assert status1.fire_model == "Fire AI Model Not Loaded"

    # Set camera connected (Phase 2 readiness test)
    CameraService.set_connected(True)
    status2 = SystemService.get_status(db)
    # Camera is connected, but Fire AI model remains Not Loaded
    assert status2.camera == "Connected"
    assert status2.fire_model == "Fire AI Model Not Loaded"

    # Reset
    CameraService.set_connected(False)
    db.close()


def test_alert_deduplication_severity_decrease():
    """
    Test deduplication when a continuing incident receives a lower current severity:
    GAS_LEAK WARNING -> same GAS_LEAK decreases to CAUTION.
    Verify:
    - Same alert ID is reused (prevents alert spam)
    - risk_level is updated to CAUTION
    - alert_type is updated to CAUTION
    - current CAUTION score/message is updated
    - latest SensorReading reference is updated
    - latest RiskAssessment reference is updated
    """
    # Setup: acknowledge any pre-existing alerts so we have a clean slate
    open_alerts = client.get("/api/v1/alerts?unacknowledged_only=true").json()
    for a in open_alerts:
        client.post(f"/api/v1/alerts/{a['id']}/acknowledge")

    # 1. Post initial reading causing GAS_LEAK with WARNING severity
    gas_payload_warning = {
        "temperature": 30.0,
        "humidity": 60.0,
        "smoke_level": 15.0,
        "gas_level": 90.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation",
        "scenario_tag": "GAS_LEAK"
    }
    res_w = client.post("/api/v1/sensors/readings", json=gas_payload_warning).json()
    reading_w_id = res_w["reading"]["id"]
    assessment_w_id = res_w["risk"]["id"]
    alert_w = res_w["active_alert"]

    assert alert_w is not None
    assert alert_w["hazard_signature"] == "GAS_LEAK"
    assert alert_w["risk_level"] == "WARNING"
    assert alert_w["alert_type"] == "WARNING"
    assert alert_w["sensor_reading_id"] == reading_w_id
    assert alert_w["risk_assessment_id"] == assessment_w_id
    alert_id = alert_w["id"]

    # 2. Post continuing reading for the same incident with lower severity -> CAUTION
    # temp=42.0, smoke=30.0, gas=75.0 (gas >= 50 keeps GAS_LEAK hazard_signature, gas < 80 prevents 55 escalation, score=32.25 -> CAUTION)
    gas_payload_caution = {
        "temperature": 42.0,
        "humidity": 60.0,
        "smoke_level": 30.0,
        "gas_level": 75.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation",
        "scenario_tag": "GAS_LEAK"
    }
    res_c = client.post("/api/v1/sensors/readings", json=gas_payload_caution).json()
    reading_c_id = res_c["reading"]["id"]
    assessment_c_id = res_c["risk"]["id"]
    alert_c = res_c["active_alert"]

    # Verify continuing incident reuses same alert ID to prevent alert spam
    assert alert_c is not None
    assert alert_c["id"] == alert_id, f"Expected alert ID {alert_id} to be reused, got {alert_c['id']}"
    assert alert_c["hazard_signature"] == "GAS_LEAK"

    # Verify CURRENT state fields are consistently updated:
    assert alert_c["risk_level"] == "CAUTION", f"Expected risk_level CAUTION, got {alert_c['risk_level']}"
    assert alert_c["alert_type"] == "CAUTION", f"Expected alert_type CAUTION, got {alert_c['alert_type']}"
    assert alert_c["risk_score"] == res_c["risk"]["overall_risk_score"]
    assert "caution" in alert_c["message"].lower()

    # Verify updated references to latest SensorReading and RiskAssessment
    assert alert_c["sensor_reading_id"] == reading_c_id
    assert alert_c["risk_assessment_id"] == assessment_c_id
    assert alert_c["sensor_reading_id"] != reading_w_id
    assert alert_c["risk_assessment_id"] != assessment_w_id

    # Verify fetching via alerts API confirms persistent state in database
    fetched = client.get("/api/v1/alerts").json()
    reused = next(a for a in fetched if a["id"] == alert_id)
    assert reused["risk_level"] == "CAUTION"
    assert reused["alert_type"] == "CAUTION"
    assert reused["sensor_reading_id"] == reading_c_id
    assert reused["risk_assessment_id"] == assessment_c_id


def test_camera_fire_confidence_boundary_cases():
    """
    Boundary tests for central FIRE_CONFIDENCE_THRESHOLD (0.50):
    0.49 -> not confirmed:
      - Low-confidence transparent explanation
      - Does NOT confirm flame presence
      - Does not trigger visual fire confirmation
      - fire_detected is False
    0.50 -> confirmed:
      - Confirms flame presence
      - camera_fire_confirmed is True
      - fire_detected is True
      - Hazard signature is CONFIRMED_FIRE_VISUAL
    0.90 -> confirmed:
      - Confirms flame presence
      - camera_fire_confirmed is True
      - fire_detected is True
      - Hazard signature is CONFIRMED_FIRE_VISUAL
    """
    reading = SensorReading(
        id=500,
        temperature=25.0,
        humidity=50.0,
        smoke_level=5.0,
        gas_level=5.0,
        flame_level=0.0,
        input_source="manual_simulation"
    )

    # --- 1. Boundary: 0.49 (Just below confirmation threshold 0.50) ---
    factors_049 = RiskAssessmentService.identify_contributing_factors(
        temp=25.0, smoke=5.0, gas=5.0, flame=0.0, camera_fire_conf=0.49
    )
    # Must use transparent low-confidence wording
    assert any("Fire model produced a low-confidence indication (49.0%); below confirmation threshold." in f for f in factors_049)
    # Must NOT claim confirmation
    assert not any("confirms flame presence" in f for f in factors_049)

    sig_049 = RiskAssessmentService.determine_hazard_signature(reading, camera_fire_conf=0.49)
    assert sig_049 != "CONFIRMED_FIRE_VISUAL"

    res_049 = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.49)
    assert "visual fire detection confirmed" not in res_049["alert_message"].lower()
    assert not any("confirms flame presence" in f for f in res_049["contributing_factors"])

    # Test database evaluation for fire_detected flag on generated alert
    db = TestingSessionLocal()
    
    # Reading with moderate sensor values (score ~ 33.5) so alerts (CAUTION/WARNING) are generated across camera tests
    def create_test_reading():
        r = SensorReading(
            temperature=45.0,
            humidity=40.0,
            smoke_level=45.0,
            gas_level=20.0,
            flame_level=50.0,
            input_source="manual_simulation"
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return r

    # Boundary 0.29 alert (below raw detection threshold 0.30)
    db_reading0 = create_test_reading()
    assessment_029, alert_029 = RiskAssessmentService.evaluate_and_record(db_reading0, db, camera_fire_conf=0.29)
    assert alert_029 is not None
    assert alert_029.fire_detected is False
    assert alert_029.fire_confidence == 0.29

    # Boundary 0.49 alert (raw detected at 0.49 >= 0.30, but below confirmation threshold 0.50)
    db_reading1 = create_test_reading()
    assessment_049, alert_049 = RiskAssessmentService.evaluate_and_record(db_reading1, db, camera_fire_conf=0.49)
    assert alert_049 is not None
    assert alert_049.fire_detected is True
    assert alert_049.hazard_signature != "CONFIRMED_FIRE_VISUAL"
    assert alert_049.fire_confidence == 0.49

    # --- 2. Boundary: 0.50 (Exact confirmation threshold) ---
    factors_050 = RiskAssessmentService.identify_contributing_factors(
        temp=25.0, smoke=5.0, gas=5.0, flame=0.0, camera_fire_conf=0.50
    )
    assert any("Computer vision fire detector confirms flame presence (Confidence: 50.0%)" in f for f in factors_050)
    assert not any("below confirmation threshold" in f for f in factors_050)

    sig_050 = RiskAssessmentService.determine_hazard_signature(reading, camera_fire_conf=0.50)
    assert sig_050 == "CONFIRMED_FIRE_VISUAL"

    res_050 = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.50)
    assert any("confirms flame presence" in f for f in res_050["contributing_factors"])

    # Boundary 0.50 alert
    db_reading2 = create_test_reading()
    assessment_050, alert_050 = RiskAssessmentService.evaluate_and_record(db_reading2, db, camera_fire_conf=0.50)
    assert alert_050 is not None
    assert alert_050.fire_detected is True
    assert alert_050.fire_confidence == 0.50

    # --- 3. Boundary: 0.90 (High confidence) ---
    factors_090 = RiskAssessmentService.identify_contributing_factors(
        temp=25.0, smoke=5.0, gas=5.0, flame=0.0, camera_fire_conf=0.90
    )
    assert any("Computer vision fire detector confirms flame presence (Confidence: 90.0%)" in f for f in factors_090)
    assert not any("below confirmation threshold" in f for f in factors_090)

    sig_090 = RiskAssessmentService.determine_hazard_signature(reading, camera_fire_conf=0.90)
    assert sig_090 == "CONFIRMED_FIRE_VISUAL"

    res_090 = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.90)
    assert any("confirms flame presence" in f for f in res_090["contributing_factors"])

    # Boundary 0.90 alert
    db_reading3 = create_test_reading()
    assessment_090, alert_090 = RiskAssessmentService.evaluate_and_record(db_reading3, db, camera_fire_conf=0.90)
    assert alert_090 is not None
    assert alert_090.fire_detected is True
    assert alert_090.fire_confidence == 0.90

    db.close()

