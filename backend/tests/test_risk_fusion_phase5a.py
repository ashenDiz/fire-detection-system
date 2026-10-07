"""
Phase 5A Multi-Modal Risk Fusion Test Suite.
Validates:
1. Fresh fire AI confidence is passed into risk evaluation.
2. Unavailable fire AI passes None.
3. Confirmed no-fire passes 0.0, NOT None.
4. Unavailable camera uses existing sensor-only 4-component renormalization (temp 0.25, smoke 0.3125, gas 0.1875, flame 0.25).
5. Available camera with 0.0 uses five-component weighting (temp 0.20, smoke 0.25, gas 0.15, flame 0.20, camera 0.20).
6. Confidence scale 0.87 is interpreted correctly as 87% (not multiplied twice to 8700%).
7. High fire confidence increases risk score according to existing 0.20 weight.
8. Visual smoke confidence does NOT replace environmental smoke_level.
9. Visual smoke does NOT add an extra numerical weight to overall risk score.
10. Unavailable person detection remains None at semantic boundary.
11. Valid zero visible people remains 0.
12. Risk bands remain unchanged (0-29 SAFE, 30-49 CAUTION, 50-69 WARNING, 70-84 HIGH RISK, 85-100 CRITICAL).
13. Phase 1 CRITICAL_FIRE sensor scenario continues to evaluate correctly.
14. Fire detection availability is persisted truthfully in RiskAssessment and Alert.
15. Valid fire confidence zero is persisted distinctly from unavailable (0.0 vs NULL, with availability boolean).
16. Code audit: no risk API or service directly calls YOLO(), model.predict(), or cv2.VideoCapture().
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.sensor import SensorReading
from app.models.risk import RiskAssessment
from app.models.alert import Alert
from app.models.event import DetectionEvent
from app.services.risk_engine import RiskAssessmentService
from app.services.fire_detection_service import FireDetectionService
from app.services.person_detection_service import PersonDetectionService
from app.services.camera_service import CameraService
from app.schemas.risk import RiskAssessmentResponse, AlertResponse, DetectionEventResponse


@pytest.fixture
def client():
    return TestClient(app)


# 1. Fresh Fire AI Confidence Passed into Risk Evaluation
def test_fresh_fire_ai_confidence_passed_into_risk_evaluation(client):
    """Verify that fresh fire AI confidence is retrieved from FireDetectionService and passed to risk engine."""
    payload = {
        "temperature": 32.0,
        "humidity": 45.0,
        "smoke_level": 10.0,
        "gas_level": 12.0,
        "flame_level": 5.0,
        "input_source": "manual_simulation",
        "scenario_tag": "TEST_PHASE5A_FRESH"
    }

    mock_summary = {
        "detection_available": True,
        "fire_detected": True,
        "smoke_detected": False,
        "fire_confidence": 0.85,
        "smoke_confidence": 0.0
    }

    with patch.object(FireDetectionService, "get_fresh_summary", return_value=mock_summary):
        with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=2):
            res = client.post("/api/v1/sensors/readings", json=payload)
            assert res.status_code == 201
            data = res.json()
            risk = data["risk"]

            assert risk["fire_detection_available"] is True
            assert risk["camera_fire_confidence"] == 0.85
            assert risk["camera_fire_risk"] == 85.0
            assert risk["people_detection_available"] is True
            assert risk["people_detected"] == 2


# 2. Unavailable Fire AI Passes None
def test_unavailable_fire_ai_passes_none(client):
    """Verify that unavailable fire detection passes None to risk evaluation and exposes null confidence."""
    payload = {
        "temperature": 25.0,
        "humidity": 50.0,
        "smoke_level": 5.0,
        "gas_level": 5.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation"
    }

    mock_summary = {
        "detection_available": False,
        "fire_detected": None,
        "smoke_detected": None,
        "fire_confidence": None,
        "smoke_confidence": None
    }

    with patch.object(FireDetectionService, "get_fresh_summary", return_value=mock_summary):
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        risk = data["risk"]

        assert risk["fire_detection_available"] is False
        assert risk["camera_fire_confidence"] is None
        assert risk["camera_fire_risk"] is None


# 3. Confirmed No-Fire Passes 0.0, NOT None
def test_confirmed_no_fire_passes_zero_not_none(client):
    """Verify that a successful inference confirming NO fire passes 0.0, preserving 5-modality fusion."""
    payload = {
        "temperature": 35.0,
        "humidity": 45.0,
        "smoke_level": 15.0,
        "gas_level": 15.0,
        "flame_level": 10.0,
        "input_source": "manual_simulation"
    }

    mock_summary = {
        "detection_available": True,
        "fire_detected": False,
        "smoke_detected": False,
        "fire_confidence": 0.0,
        "smoke_confidence": 0.0
    }

    with patch.object(FireDetectionService, "get_fresh_summary", return_value=mock_summary):
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        risk = data["risk"]

        assert risk["fire_detection_available"] is True
        assert risk["camera_fire_confidence"] == 0.0
        assert risk["camera_fire_risk"] == 0.0
        assert risk["camera_fire_confidence"] is not None


# 4. Unavailable Camera Uses Existing 4-Component Renormalization vs 5. 5-Component Weighting
def test_unavailable_camera_vs_confirmed_no_fire_weighting_difference():
    """
    Mathematical proof of distinct fusion behavior:
    When camera is unavailable: weights are renormalized over 4 sensors (sum=0.80):
      temp = 0.20/0.80 = 0.25
      smoke = 0.25/0.80 = 0.3125
      gas = 0.15/0.80 = 0.1875
      flame = 0.20/0.80 = 0.25
    When camera is available with confirmed 0.0 fire:
      weights sum to 1.0 (camera has 0.20 weight with contribution 0.0),
      reducing the overall score by 20% compared to 4-sensor renormalized score.
    """
    reading = SensorReading(
        temperature=40.0,  # elevated temp -> subscore > 0
        humidity=45.0,
        smoke_level=30.0,  # elevated smoke -> subscore > 0
        gas_level=30.0,    # elevated gas -> subscore > 0
        flame_level=30.0,  # elevated flame -> subscore > 0
        input_source="manual_simulation"
    )

    # 4a. Camera unavailable
    res_unavailable = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None)
    assert res_unavailable["fire_detection_available"] is False
    assert res_unavailable["camera_fire_risk"] is None

    # 4b. Camera confirmed NO FIRE (0.0)
    res_zero_fire = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.0)
    assert res_zero_fire["fire_detection_available"] is True
    assert res_zero_fire["camera_fire_risk"] == 0.0

    # The score with confirmed no fire must be lower than with unavailable camera
    # because the camera actively confirms no combustion flame
    assert res_zero_fire["overall_risk_score"] < res_unavailable["overall_risk_score"]

    # Verify exact mathematical relation: score_zero = score_unavailable * 0.80
    expected_ratio = res_zero_fire["overall_risk_score"] / res_unavailable["overall_risk_score"]
    assert pytest.approx(expected_ratio, rel=1e-2) == 0.80


# 6. Confidence Scale 0.87 Interpreted Correctly (Not Multiplied Twice)
def test_confidence_scale_0_87_not_multiplied_twice():
    """Verify confidence 0.87 contributes 87.0 risk points (87%), NOT 8700%."""
    reading = SensorReading(
        temperature=20.0,
        humidity=50.0,
        smoke_level=0.0,
        gas_level=0.0,
        flame_level=0.0,
        input_source="manual_simulation"
    )

    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.87)

    assert result["camera_fire_risk"] == 87.0
    assert result["camera_fire_confidence"] == 0.87
    # Overall score contribution from camera: 87.0 * 0.20 = 17.4
    assert result["overall_risk_score"] == 17.4


# 7. High Fire Confidence Increases Risk According to Existing 0.20 Weight
def test_high_fire_confidence_increases_risk_by_0_20_weight():
    """Verify that a delta in fire confidence produces exact delta * 0.20 in overall risk score."""
    reading = SensorReading(
        temperature=20.0,
        humidity=50.0,
        smoke_level=0.0,
        gas_level=0.0,
        flame_level=0.0,
        input_source="manual_simulation"
    )

    res_50 = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.50)
    res_90 = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.90)

    # 50% conf -> 50.0 * 0.20 = 10.0
    # 90% conf -> 90.0 * 0.20 = 18.0
    assert res_50["overall_risk_score"] == 10.0
    assert res_90["overall_risk_score"] == 18.0
    assert pytest.approx(res_90["overall_risk_score"] - res_50["overall_risk_score"], abs=0.01) == 8.0


# 8. Visual Smoke Confidence Does NOT Replace Environmental smoke_level
def test_visual_smoke_does_not_replace_environmental_smoke():
    """Verify that visual smoke detection never overwrites or alters environmental sensor reading."""
    reading = SensorReading(
        temperature=25.0,
        humidity=50.0,
        smoke_level=8.0,  # low environmental smoke
        gas_level=5.0,
        flame_level=0.0,
        input_source="manual_simulation"
    )

    result = RiskAssessmentService.assess_risk(
        reading,
        camera_fire_conf=0.0,
        visual_smoke_detected=True,
        visual_smoke_conf=0.95
    )

    # Environmental smoke must remain unchanged
    assert reading.smoke_level == 8.0
    # Smoke sub-score must be calculated from 8.0, NOT 95.0
    assert result["smoke_risk"] == pytest.approx(8.0, abs=1.0)


# 9. Visual Smoke Does NOT Add an Extra Numerical Weight
def test_visual_smoke_does_not_add_extra_numerical_weight():
    """Verify visual smoke confidence does not alter overall numerical risk score in Phase 5A."""
    reading = SensorReading(
        temperature=35.0,
        humidity=40.0,
        smoke_level=20.0,
        gas_level=15.0,
        flame_level=10.0,
        input_source="manual_simulation"
    )

    res_without_smoke = RiskAssessmentService.assess_risk(
        reading,
        camera_fire_conf=0.75,
        visual_smoke_detected=False,
        visual_smoke_conf=0.0
    )

    res_with_smoke = RiskAssessmentService.assess_risk(
        reading,
        camera_fire_conf=0.75,
        visual_smoke_detected=True,
        visual_smoke_conf=0.98
    )

    # Must produce the exact same numerical overall risk score
    assert res_without_smoke["overall_risk_score"] == res_with_smoke["overall_risk_score"]


# 10. Unavailable Person Detection Remains None at Semantic Boundary & 11. Valid Zero Remains 0
def test_person_detection_semantics_preserved(client):
    """Verify PersonDetectionService unavailable vs fresh zero semantics are preserved."""
    payload = {
        "temperature": 25.0,
        "humidity": 50.0,
        "smoke_level": 5.0,
        "gas_level": 5.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation"
    }

    # Case A: Unavailable person detection
    with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=None):
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["risk"]["people_detection_available"] is False
        assert data["risk"]["people_detected"] is None

    # Case B: Valid zero visible people
    with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=0):
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["risk"]["people_detection_available"] is True
        assert data["risk"]["people_detected"] == 0


# 12. Risk Bands Remain Unchanged
def test_risk_bands_remain_unchanged():
    """Verify standard operational risk bands: 0-29 SAFE, 30-49 CAUTION, 50-69 WARNING, 70-84 HIGH RISK, 85-100 CRITICAL."""
    assert RiskAssessmentService.determine_risk_level(0.0) == "SAFE"
    assert RiskAssessmentService.determine_risk_level(29.99) == "SAFE"
    assert RiskAssessmentService.determine_risk_level(30.0) == "CAUTION"
    assert RiskAssessmentService.determine_risk_level(49.99) == "CAUTION"
    assert RiskAssessmentService.determine_risk_level(50.0) == "WARNING"
    assert RiskAssessmentService.determine_risk_level(69.99) == "WARNING"
    assert RiskAssessmentService.determine_risk_level(70.0) == "HIGH RISK"
    assert RiskAssessmentService.determine_risk_level(84.99) == "HIGH RISK"
    assert RiskAssessmentService.determine_risk_level(85.0) == "CRITICAL"
    assert RiskAssessmentService.determine_risk_level(100.0) == "CRITICAL"


# 13. Phase 1 CRITICAL_FIRE Scenario Behaves Correctly
def test_critical_fire_scenario_evaluation():
    """Verify severe fire scenario escalates to CRITICAL both with and without visual AI."""
    reading = SensorReading(
        temperature=75.0,
        humidity=20.0,
        smoke_level=95.0,
        gas_level=70.0,
        flame_level=90.0,
        input_source="manual_simulation"
    )

    # Sensor-only (camera unavailable)
    res_sensor_only = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None)
    assert res_sensor_only["risk_level"] == "CRITICAL"
    assert res_sensor_only["overall_risk_score"] >= 85.0

    # With high camera fire confidence
    res_with_cam = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.95)
    assert res_with_cam["risk_level"] == "CRITICAL"
    assert res_with_cam["overall_risk_score"] >= 85.0


# 14. Fire Detection Availability is Persisted Truthfully & 15. Valid Zero vs Unavailable Distinct in DB
def test_fire_detection_availability_and_zero_persisted_distinctly(db_session):
    """Verify RiskAssessment and Alert truthful storage of unavailable (False, NULL) vs zero (True, 0.0)."""
    reading = SensorReading(
        temperature=30.0,
        humidity=50.0,
        smoke_level=10.0,
        gas_level=10.0,
        flame_level=5.0,
        input_source="manual_simulation"
    )
    db_session.add(reading)
    db_session.commit()
    db_session.refresh(reading)

    # 1. Unavailable camera
    a_unavail, _ = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=None
    )
    assert a_unavail.fire_detection_available is False
    assert a_unavail.fire_confidence is None
    assert a_unavail.camera_fire_risk is None

    # 2. Confirmed NO FIRE (0.0)
    a_zero, _ = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=0.0
    )
    assert a_zero.fire_detection_available is True
    assert a_zero.fire_confidence == 0.0
    assert a_zero.camera_fire_risk == 0.0

    # 3. Fire detected with confidence 0.88
    a_fire, alert = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=0.88
    )
    assert a_fire.fire_detection_available is True
    assert a_fire.fire_confidence == 0.88
    assert a_fire.camera_fire_risk == 88.0

    # In Alert record
    if alert is not None:
        assert alert.fire_detection_available is True
        assert alert.fire_detected is True
        assert alert.fire_confidence == 0.88


# 16. Code Audit: No Prohibited Hardware/Model Calls in Risk Engine or Sensor API
def test_code_audit_no_direct_inference_or_camera_in_risk_logic():
    """Verify that RiskAssessmentService and sensors API never create VideoCapture or invoke YOLO directly."""
    import inspect
    import app.services.risk_engine as risk_engine_module
    import app.api.v1.sensors as sensors_module
    import app.api.v1.risk as risk_module

    risk_engine_source = inspect.getsource(risk_engine_module)
    sensors_source = inspect.getsource(sensors_module)
    risk_api_source = inspect.getsource(risk_module)

    combined_sources = risk_engine_source + sensors_source + risk_api_source

    prohibited_patterns = [
        "cv2.VideoCapture(",
        "YOLO(",
        "model.predict(",
        "source=0",
        "random.random",
        "random.uniform"
    ]

    for pattern in prohibited_patterns:
        assert pattern not in combined_sources, f"PROHIBITED PATTERN '{pattern}' found in risk/sensor production source!"


# 17. DetectionEvent with Unavailable Fire AI Persists fire_detection_available=False
def test_detection_event_persists_unavailable_fire(db_session):
    """Verify DetectionEvent persists fire_detection_available=False and fire_confidence=None when camera unavailable."""
    reading = SensorReading(
        temperature=60.0,
        humidity=40.0,
        smoke_level=55.0,
        gas_level=20.0,
        flame_level=60.0,
        input_source="manual_simulation"
    )
    db_session.add(reading)
    db_session.commit()
    db_session.refresh(reading)

    assessment, alert = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=None
    )

    event = db_session.query(DetectionEvent).order_by(DetectionEvent.created_at.desc()).first()
    assert event is not None
    assert event.fire_detection_available is False
    assert event.fire_detected is False  # DB storage sentinel
    assert event.fire_confidence is None


# 18. DetectionEvent with Confirmed Fresh No-Fire Persists fire_detection_available=True, fire_detected=False
def test_detection_event_persists_confirmed_no_fire(db_session):
    """Verify DetectionEvent persists fire_detection_available=True, fire_detected=False, fire_confidence=0.0."""
    reading = SensorReading(
        temperature=65.0,
        humidity=35.0,
        smoke_level=60.0,
        gas_level=20.0,
        flame_level=65.0,
        input_source="manual_simulation"
    )
    db_session.add(reading)
    db_session.commit()
    db_session.refresh(reading)

    assessment, alert = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=0.0
    )

    event = db_session.query(DetectionEvent).order_by(DetectionEvent.created_at.desc()).first()
    assert event is not None
    assert event.fire_detection_available is True
    assert event.fire_detected is False
    assert event.fire_confidence == 0.0


# 19. DetectionEvent with Detected Fire Persists fire_detection_available=True, fire_detected=True
def test_detection_event_persists_detected_fire(db_session):
    """Verify DetectionEvent persists fire_detection_available=True, fire_detected=True, and actual confidence."""
    reading = SensorReading(
        temperature=70.0,
        humidity=30.0,
        smoke_level=70.0,
        gas_level=20.0,
        flame_level=70.0,
        input_source="manual_simulation"
    )
    db_session.add(reading)
    db_session.commit()
    db_session.refresh(reading)

    assessment, alert = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_session,
        camera_fire_conf=0.89
    )

    event = db_session.query(DetectionEvent).order_by(DetectionEvent.created_at.desc()).first()
    assert event is not None
    assert event.fire_detection_available is True
    assert event.fire_detected is True
    assert event.fire_confidence == 0.89


# 20. DetectionEventResponse Maps Unavailable DB False Sentinel to None
def test_detection_event_response_normalizes_unavailable():
    """Verify DetectionEventResponse maps fire_detection_available=False to fire_detected=None and fire_confidence=None."""
    # Simulating ORM model with DB storage sentinel
    event = DetectionEvent(
        id=10,
        fire_detection_available=False,
        fire_detected=False,  # DB sentinel
        fire_confidence=None,
        people_count=0,
        people_detection_available=False,
        risk_score=60.0,
        risk_level="WARNING",
        camera_status="Disconnected",
        created_at=datetime.now(timezone.utc)
    )

    resp = DetectionEventResponse.model_validate(event)
    assert resp.fire_detection_available is False
    assert resp.fire_detected is None
    assert resp.fire_confidence is None
    assert resp.people_count is None


# 21. AlertResponse Maps Unavailable DB False Sentinel to None
def test_alert_response_normalizes_unavailable():
    """Verify AlertResponse maps fire_detection_available=False to fire_detected=None and fire_confidence=None."""
    alert = Alert(
        id=20,
        hazard_signature="GAS_LEAK",
        risk_score=65.0,
        risk_level="WARNING",
        alert_type="WARNING",
        message="Gas leak detected",
        people_detected=0,
        people_detection_available=False,
        fire_detection_available=False,
        fire_detected=False,  # DB sentinel
        fire_confidence=None,
        acknowledged=False,
        created_at=datetime.now(timezone.utc)
    )

    resp = AlertResponse.model_validate(alert)
    assert resp.fire_detection_available is False
    assert resp.fire_detected is None
    assert resp.fire_confidence is None
    assert resp.people_detected is None


# 22. AlertResponse Preserves Fresh Confirmed No-Fire as False and 0.0
def test_alert_response_preserves_confirmed_no_fire():
    """Verify AlertResponse preserves fire_detection_available=True with fire_detected=False and fire_confidence=0.0."""
    alert = Alert(
        id=21,
        hazard_signature="HIGH_HEAT",
        risk_score=75.0,
        risk_level="HIGH RISK",
        alert_type="HIGH",
        message="High heat without flame",
        people_detected=2,
        people_detection_available=True,
        fire_detection_available=True,
        fire_detected=False,  # Confirmed NO FIRE
        fire_confidence=0.0,
        acknowledged=False,
        created_at=datetime.now(timezone.utc)
    )

    resp = AlertResponse.model_validate(alert)
    assert resp.fire_detection_available is True
    assert resp.fire_detected is False
    assert resp.fire_confidence == 0.0
    assert resp.people_detected == 2


# 23. Camera Generation Unchanged: Normal Person + Fire Semantic Values Used
def test_camera_generation_unchanged_accepts_snapshot(client):
    """Verify that when CameraService generation is stable before and after, AI snapshot is accepted."""
    payload = {
        "temperature": 28.0,
        "humidity": 50.0,
        "smoke_level": 8.0,
        "gas_level": 10.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation"
    }

    mock_summary = {
        "detection_available": True,
        "fire_detected": True,
        "smoke_detected": False,
        "fire_confidence": 0.85,
        "smoke_confidence": 0.0
    }

    with patch.object(CameraService, "get_generation", return_value=7):
        with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=3):
            with patch.object(FireDetectionService, "get_fresh_summary", return_value=mock_summary):
                res = client.post("/api/v1/sensors/readings", json=payload)
                assert res.status_code == 201
                data = res.json()["risk"]
                assert data["people_detection_available"] is True
                assert data["people_detected"] == 3
                assert data["fire_detection_available"] is True
                assert data["camera_fire_confidence"] == 0.85


# 24. Camera Generation Changes Between Person/Fire Snapshot: Both Treated as Unavailable
def test_camera_generation_change_discards_incoherent_snapshot(client):
    """
    Verify that if CameraService generation changes between person and fire snapshot collection,
    both modalities are safely treated as unavailable to prevent session tearing.
    """
    payload = {
        "temperature": 28.0,
        "humidity": 50.0,
        "smoke_level": 8.0,
        "gas_level": 10.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation"
    }

    mock_summary = {
        "detection_available": True,
        "fire_detected": True,
        "smoke_detected": True,
        "fire_confidence": 0.90,
        "smoke_confidence": 0.85
    }

    # Generation before is 7, generation after is 8 (camera restarted during acquisition)
    with patch.object(CameraService, "get_generation", side_effect=[7, 8]):
        with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=3):
            with patch.object(FireDetectionService, "get_fresh_summary", return_value=mock_summary):
                res = client.post("/api/v1/sensors/readings", json=payload)
                assert res.status_code == 201
                data = res.json()["risk"]
                # Must be discarded and marked unavailable
                assert data["people_detection_available"] is False
                assert data["people_detected"] is None
                assert data["fire_detection_available"] is False
                assert data["camera_fire_confidence"] is None
                assert data["camera_fire_risk"] is None
