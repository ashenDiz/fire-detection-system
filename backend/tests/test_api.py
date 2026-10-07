"""
Integration and validation tests for the FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_system_status_endpoint():
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "Online"
    assert data["database"] == "Connected"
    assert data["sensor_mode"] == "Manual Simulation"
    assert "timestamp" in data


def test_list_scenarios_endpoint():
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) >= 4
    scenario_ids = [s["id"] for s in scenarios]
    assert "NORMAL" in scenario_ids
    assert "GAS_LEAK" in scenario_ids
    assert "POSSIBLE_FIRE" in scenario_ids
    assert "CRITICAL_FIRE" in scenario_ids


def test_submit_valid_sensor_reading():
    payload = {
        "temperature": 29.5,
        "humidity": 58.0,
        "flame_level": 0.0,
        "smoke_level": 4.0,
        "gas_level": 5.0,
        "input_source": "manual_simulation",
        "scenario_tag": "TEST_RUN"
    }
    response = client.post("/api/v1/sensors/readings", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "reading" in data
    assert "risk" in data
    assert data["reading"]["temperature"] == 29.5
    assert data["reading"]["input_source"] == "manual_simulation"
    assert data["risk"]["risk_level"] == "SAFE"


def test_submit_invalid_sensor_ranges():
    # Out of range temperature (> 120)
    res_temp = client.post("/api/v1/sensors/readings", json={
        "temperature": 150.0,
        "humidity": 50.0,
        "flame_level": 0.0,
        "smoke_level": 0.0,
        "gas_level": 0.0,
    })
    assert res_temp.status_code == 422

    # Negative humidity
    res_hum = client.post("/api/v1/sensors/readings", json={
        "temperature": 25.0,
        "humidity": -5.0,
        "flame_level": 0.0,
        "smoke_level": 0.0,
        "gas_level": 0.0,
    })
    assert res_hum.status_code == 422

    # Out of range smoke (> 100)
    res_smoke = client.post("/api/v1/sensors/readings", json={
        "temperature": 25.0,
        "humidity": 50.0,
        "flame_level": 0.0,
        "smoke_level": 110.0,
        "gas_level": 0.0,
    })
    assert res_smoke.status_code == 422


def test_get_latest_reading_and_risk():
    # Submit a reading
    client.post("/api/v1/sensors/readings", json={
        "temperature": 75.0,
        "humidity": 25.0,
        "flame_level": 90.0,
        "smoke_level": 95.0,
        "gas_level": 70.0,
        "input_source": "manual_simulation",
        "scenario_tag": "CRITICAL_FIRE"
    })

    # Get latest sensor
    res_latest = client.get("/api/v1/sensors/latest")
    assert res_latest.status_code == 200
    latest_data = res_latest.json()
    assert latest_data["temperature"] == 75.0

    # Get current risk
    res_risk = client.get("/api/v1/risk/current")
    assert res_risk.status_code == 200
    risk_data = res_risk.json()
    assert risk_data["risk_level"] in ["HIGH RISK", "CRITICAL"]
    assert len(risk_data["contributing_factors"]) > 0


def test_alerts_generated_traceability_and_acknowledgement():
    # Acknowledge any pre-existing unacknowledged alerts to ensure clean state
    existing_alerts = client.get("/api/v1/alerts?unacknowledged_only=true").json()
    for a in existing_alerts:
        client.post(f"/api/v1/alerts/{a['id']}/acknowledge")

    # Trigger a new alert by posting high hazard data
    post_res = client.post("/api/v1/sensors/readings", json={
        "temperature": 80.0,
        "humidity": 20.0,
        "flame_level": 95.0,
        "smoke_level": 95.0,
        "gas_level": 80.0,
        "input_source": "manual_simulation",
        "scenario_tag": "ALERT_TEST"
    })

    assert post_res.status_code == 201
    reading_id = post_res.json()["reading"]["id"]
    risk_id = post_res.json()["risk"]["id"]
    alert_info = post_res.json().get("active_alert")
    assert alert_info is not None
    alert_id = alert_info["id"]

    # Verify foreign key traceability
    assert alert_info["sensor_reading_id"] == reading_id
    assert alert_info["risk_assessment_id"] == risk_id

    # Verify AI fire detection is truthfully None when camera is inactive
    assert alert_info["fire_detected"] is None
    assert alert_info["fire_detection_available"] is False
    assert alert_info["fire_confidence"] is None

    # Retrieve alerts list
    res_alerts = client.get("/api/v1/alerts")
    assert res_alerts.status_code == 200
    alerts_list = res_alerts.json()
    assert any(a["id"] == alert_id for a in alerts_list)

    # Acknowledge the alert
    ack_res = client.post(f"/api/v1/alerts/{alert_id}/acknowledge")
    assert ack_res.status_code == 200
    assert ack_res.json()["acknowledged"] is True
    assert ack_res.json()["acknowledged_at"] is not None


def test_alert_deduplication_on_repeated_dangerous_readings():
    """Verify that repeated identical dangerous readings do not spam duplicate unacknowledged alerts."""
    dangerous_payload = {
        "temperature": 30.0,
        "humidity": 50.0,
        "smoke_level": 15.0,
        "gas_level": 90.0,
        "flame_level": 0.0,
        "input_source": "manual_simulation",
        "scenario_tag": "GAS_LEAK"
    }

    # Submit 3 times consecutively
    res1 = client.post("/api/v1/sensors/readings", json=dangerous_payload).json()
    alert1 = res1.get("active_alert")
    assert alert1 is not None

    res2 = client.post("/api/v1/sensors/readings", json=dangerous_payload).json()
    alert2 = res2.get("active_alert")
    assert alert2 is not None
    assert alert2["id"] == alert1["id"], "Identical hazard state should reuse active unacknowledged alert"

    res3 = client.post("/api/v1/sensors/readings", json=dangerous_payload).json()
    alert3 = res3.get("active_alert")
    assert alert3 is not None
    assert alert3["id"] == alert1["id"], "Alert ID should remain the same without spamming new rows"


def test_detection_events_logged_and_retrieved():
    """Verify DetectionEvent is logged on state changes and accessible via GET /api/v1/events."""
    # First submit a normal reading to set a baseline state
    client.post("/api/v1/sensors/readings", json={
        "temperature": 25.0,
        "humidity": 55.0,
        "flame_level": 0.0,
        "smoke_level": 2.0,
        "gas_level": 3.0,
        "input_source": "manual_simulation",
        "scenario_tag": "NORMAL"
    })

    # Transition to CRITICAL state (state change trigger)
    client.post("/api/v1/sensors/readings", json={
        "temperature": 75.0,
        "humidity": 25.0,
        "flame_level": 95.0,
        "smoke_level": 95.0,
        "gas_level": 70.0,
        "input_source": "manual_simulation",
        "scenario_tag": "CRITICAL_FIRE"
    })

    # Query events endpoint
    events_res = client.get("/api/v1/events")
    assert events_res.status_code == 200
    events = events_res.json()
    assert len(events) > 0
    latest_event = events[0]
    assert "risk_score" in latest_event
    assert "risk_level" in latest_event
    assert latest_event["camera_status"] == "Disconnected"
