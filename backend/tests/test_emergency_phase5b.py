"""
Phase 5B Emergency Response, Temporal Confirmation, and Email Notification Test Suite.
Hardened for Phase 5B with:
1. Unique fire inference identity tracking (rejects duplicate monitor poll cycles).
2. Source-aware emergency clearing (SENSOR_CRITICAL cleared by newer non-critical telemetry; VISUAL requires 5 zeros).
3. Strict timestamp validation (missing timestamps are never considered fresh).
4. Reliable .env loading and environment override support.
5. SMTP authentication requirement settings and credential safety.
6. Background monitor worker lifecycle tests (no duplicate workers, clean restart).
7. Deterministic mock SMTP tests (anti-spam episode guard, failure isolation).
8. Architectural boundaries (no YOLO or VideoCapture in emergency/email services).
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings, ENV_FILE_PATH
from app.main import app
from app.services.camera_service import CameraService
from app.services.fire_detection_service import FireDetectionService
from app.services.emergency_response_service import EmergencyResponseService
from app.services.email_notification_service import EmailNotificationService


@pytest.fixture(autouse=True)
def reset_emergency_state():
    """Reset EmergencyResponseService state before every test."""
    EmergencyResponseService.reset_state()
    yield
    EmergencyResponseService.reset_state()


@pytest.fixture
def client():
    return TestClient(app)


# --- SECTION 1 & 2: THRESHOLD SEPARATION & UNIQUE INFERENCE IDENTITIES ---

def test_029_raw_candidate_not_detected():
    """1. 0.29: raw fire candidate not detected; remains NORMAL."""
    fire_summary = {
        "detection_available": True,
        "fire_detected": False,
        "fire_confidence": 0.29,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 1,
    }
    status = EmergencyResponseService.evaluate_step(fire_summary, current_timestamp=100.0)
    assert status["state"] == "NORMAL"
    assert status["temporal_hits"] == 0
    assert status["priority"] == "LOW"


def test_035_raw_candidate_triggers_monitoring_not_emergency():
    """2. 0.35: raw visual candidate triggers MONITORING, NOT confirmed emergency."""
    fire_summary = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.35,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 2,
    }
    status = EmergencyResponseService.evaluate_step(fire_summary, current_timestamp=100.0)
    assert status["state"] == "MONITORING"
    assert "Possible visual fire detected" in status["reason"]
    assert status["temporal_hits"] == 0


def test_single_055_hit_does_not_confirm():
    """3. Single 0.55 hit does NOT confirm emergency when MIN_HITS is 2."""
    fire_summary = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.55,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 3,
    }
    status = EmergencyResponseService.evaluate_step(fire_summary, current_timestamp=100.0)
    assert status["temporal_hits"] == 1
    assert status["state"] == "MONITORING"


def test_two_strong_hits_inside_window_confirms_emergency():
    """4. Two DISTINCT >=0.50 fresh hits inside 2.0 sec confirms emergency."""
    fire_1 = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.52,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 10,
    }
    fire_2 = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.61,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 11,
    }
    # Hit 1 at t=100.0
    status_1 = EmergencyResponseService.evaluate_step(fire_1, current_timestamp=100.0)
    assert status_1["state"] == "MONITORING"
    assert status_1["temporal_hits"] == 1

    # Hit 2 at t=101.2 (within 2.0s window, distinct inference identity)
    status_2 = EmergencyResponseService.evaluate_step(fire_2, current_timestamp=101.2)
    assert status_2["state"] == "EMERGENCY_CONFIRMED"
    assert status_2["temporal_hits"] == 2
    assert status_2["confirmation_source"] == "VISUAL_TEMPORAL"
    assert "Visual fire emergency confirmed" in status_2["reason"]


def test_two_strong_hits_outside_window_do_not_confirm():
    """5. Two >=0.50 hits outside the 2.0 sec time window do NOT confirm emergency."""
    fire_1 = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.55,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 20,
    }
    fire_2 = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.58,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 21,
    }
    # Hit 1 at t=100.0
    EmergencyResponseService.evaluate_step(fire_1, current_timestamp=100.0)

    # Hit 2 at t=103.0 (3.0 seconds later > 2.0s window)
    status_2 = EmergencyResponseService.evaluate_step(fire_2, current_timestamp=103.0)
    assert status_2["temporal_hits"] == 1  # Previous hit expired
    assert status_2["state"] == "MONITORING"


def test_same_strong_inference_polled_10_times_rejects_duplicates():
    """A) Same strong inference identity polled 10 times -> temporal_hits remains exactly 1 -> NOT confirmed."""
    cached_fire = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.85,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 999,  # Same inference identity
    }
    for poll in range(10):
        t = 100.0 + (poll * 0.1)  # Advance time by 100ms each poll cycle
        status = EmergencyResponseService.evaluate_step(cached_fire, current_timestamp=t)
        assert status["temporal_hits"] == 1, f"Duplicate counted on poll {poll+1}"
        assert status["state"] == "MONITORING", f"Premature confirmation on poll {poll+1}"


def test_same_fresh_zero_polled_10_times_does_not_clear():
    """C) Same fresh-zero inference identity polled 10 times -> negative counter only 1 -> does NOT clear."""
    # Confirm visual emergency first
    f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 1}
    f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 2}
    EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
    EmergencyResponseService.evaluate_step(f2, current_timestamp=100.5)
    assert EmergencyResponseService.get_status()["state"] == "EMERGENCY_CONFIRMED"

    # Poll 10 times with the EXACT SAME zero inference
    cached_zero = {
        "detection_available": True,
        "fire_detected": False,
        "fire_confidence": 0.0,
        "inference_id": 50,  # Constant inference identity
    }
    for poll in range(10):
        t = 101.0 + (poll * 0.1)
        st = EmergencyResponseService.evaluate_step(cached_zero, current_timestamp=t)
        assert st["state"] == "EMERGENCY_CONFIRMED", f"Cleared prematurely on poll {poll+1}"


def test_five_distinct_fresh_zeros_clears_visual_emergency():
    """D) 5 DISTINCT successful fresh-zero inference identities clear visual emergency."""
    # Confirm visual emergency first
    f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 1}
    f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 2}
    EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
    EmergencyResponseService.evaluate_step(f2, current_timestamp=100.5)
    assert EmergencyResponseService.get_status()["state"] == "EMERGENCY_CONFIRMED"

    # Provide 4 distinct fresh zeros
    for i in range(1, 5):
        zero_sample = {
            "detection_available": True,
            "fire_detected": False,
            "fire_confidence": 0.0,
            "inference_id": 100 + i,
        }
        st = EmergencyResponseService.evaluate_step(zero_sample, current_timestamp=101.0 + i)
        assert st["state"] == "EMERGENCY_CONFIRMED"

    # 5th distinct fresh zero -> must clear
    zero_5 = {
        "detection_available": True,
        "fire_detected": False,
        "fire_confidence": 0.0,
        "inference_id": 105,
    }
    st5 = EmergencyResponseService.evaluate_step(zero_5, current_timestamp=106.0)
    assert st5["state"] == "NORMAL"
    assert "cleared" in st5["reason"].lower()


def test_unavailable_inference_never_counts_as_temporal_or_clearing():
    """E) Unavailable inference never counts as temporal hit and never clears emergency."""
    # In NORMAL state
    unavail = {"detection_available": False, "fire_detected": None, "fire_confidence": None, "inference_id": None}
    st = EmergencyResponseService.evaluate_step(unavail, current_timestamp=100.0)
    assert st["state"] == "NORMAL"
    assert st["temporal_hits"] == 0

    # In EMERGENCY_CONFIRMED state
    f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 1}
    f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 2}
    EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
    EmergencyResponseService.evaluate_step(f2, current_timestamp=100.5)
    assert EmergencyResponseService.get_status()["state"] == "EMERGENCY_CONFIRMED"

    # 10 unavailable samples must NOT clear
    for i in range(10):
        st = EmergencyResponseService.evaluate_step(unavail, current_timestamp=101.0 + i)
        assert st["state"] == "EMERGENCY_CONFIRMED"


# --- SECTION 26 & 3: MULTI-MODAL & SOURCE-AWARE EMERGENCY CLEARING ---

class DummySensorReading:
    def __init__(self, temp=25.0, smoke=5.0, gas=5.0, flame=0.0, age_sec=0.0):
        self.temperature = temp
        self.smoke_level = smoke
        self.gas_level = gas
        self.flame_level = flame
        self.input_source = "manual_simulation"
        if age_sec is not None:
            self.created_at = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
        else:
            self.created_at = None


def test_candidate_plus_recent_flame_confirms():
    """Visual candidate >=0.30 + recent strong flame reading confirms emergency."""
    fire_cand = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.35,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 501,
    }
    reading = DummySensorReading(flame=65.0, age_sec=5.0)  # flame >= 60, recent 5s
    status = EmergencyResponseService.evaluate_step(fire_cand, sensor_reading=reading)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["confirmation_source"] == "MULTIMODAL"
    assert "corroborated by recent simulated environmental telemetry" in status["reason"]


def test_candidate_plus_recent_smoke_and_temp_confirms():
    """Visual candidate >=0.30 + recent smoke >=40 and temp >=50 confirms emergency."""
    fire_cand = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.38,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 502,
    }
    reading = DummySensorReading(smoke=45.0, temp=55.0, age_sec=10.0)
    status = EmergencyResponseService.evaluate_step(fire_cand, sensor_reading=reading)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["confirmation_source"] == "MULTIMODAL"


def test_candidate_plus_stale_reading_does_not_corroborate():
    """Visual candidate >=0.30 + STALE reading (>30s) does NOT confirm emergency."""
    fire_cand = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.35,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 503,
    }
    stale_reading = DummySensorReading(flame=90.0, age_sec=45.0)  # age 45s > 30s timeout
    status = EmergencyResponseService.evaluate_step(fire_cand, sensor_reading=stale_reading)
    assert status["state"] == "MONITORING"
    assert status["sensor_context_available"] is False


def test_sensor_critical_emergency_latching_and_clearing():
    """
    Source-aware clearing for SENSOR_CRITICAL:
    Camera unavailable + recent CRITICAL risk -> SENSOR_CRITICAL emergency.
    Stale / no sensor update -> remains emergency.
    Newer recent non-critical risk assessment -> clears to NORMAL without requiring camera.
    """
    unavail_camera = {"detection_available": False, "fire_detected": None, "fire_confidence": None}

    crit_time = datetime.now(timezone.utc) - timedelta(seconds=5.0)
    crit_assessment = MagicMock(risk_level="CRITICAL", created_at=crit_time)

    # 1. Trigger SENSOR_CRITICAL emergency
    status = EmergencyResponseService.evaluate_step(unavail_camera, risk_assessment=crit_assessment)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["confirmation_source"] == "SENSOR_CRITICAL"

    # 2. Stale assessment (>30s) -> must NOT clear emergency!
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=40.0)
    stale_assessment = MagicMock(risk_level="SAFE", created_at=stale_time)
    stale_status = EmergencyResponseService.evaluate_step(unavail_camera, risk_assessment=stale_assessment)
    assert stale_status["state"] == "EMERGENCY_CONFIRMED", "Stale telemetry must not clear emergency"

    # 3. No sensor update (None) -> must NOT clear emergency!
    no_update_status = EmergencyResponseService.evaluate_step(unavail_camera, risk_assessment=None)
    assert no_update_status["state"] == "EMERGENCY_CONFIRMED"

    # 4. Newer recent non-critical assessment -> clears to NORMAL
    newer_time = datetime.now(timezone.utc) - timedelta(seconds=1.0)
    safe_assessment = MagicMock(risk_level="SAFE", created_at=newer_time)
    cleared_status = EmergencyResponseService.evaluate_step(unavail_camera, risk_assessment=safe_assessment)
    assert cleared_status["state"] == "NORMAL"
    assert cleared_status["confirmation_source"] is None


def test_visual_smoke_does_not_alter_numerical_risk_weights():
    """Visual smoke must NOT become an additional Phase 5A risk weight."""
    w_sum = (
        settings.WEIGHT_TEMPERATURE +
        settings.WEIGHT_SMOKE +
        settings.WEIGHT_GAS +
        settings.WEIGHT_FLAME +
        settings.WEIGHT_CAMERA_FIRE
    )
    assert round(w_sum, 2) == 1.0
    assert hasattr(settings, "WEIGHT_CAMERA_FIRE")
    assert not hasattr(settings, "WEIGHT_CAMERA_SMOKE")
    assert not hasattr(settings, "WEIGHT_VISUAL_SMOKE")


# --- SECTION 4: MISSING TIMESTAMPS MUST NOT BE CONSIDERED FRESH ---

def test_missing_sensor_reading_timestamp_not_considered_fresh():
    """Missing sensor_reading.created_at -> sensor_context_available=False -> cannot corroborate fire."""
    fire_cand = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.35,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 601,
    }
    reading_undated = DummySensorReading(flame=85.0, age_sec=None)  # created_at is None
    status = EmergencyResponseService.evaluate_step(fire_cand, sensor_reading=reading_undated)
    assert status["sensor_context_available"] is False
    assert status["state"] == "MONITORING"


def test_missing_risk_assessment_timestamp_cannot_qualify_as_critical():
    """Missing risk_assessment.created_at -> cannot qualify as recent sensor-critical evidence."""
    unavail_camera = {"detection_available": False, "fire_detected": None, "fire_confidence": None}
    assessment_undated = MagicMock(risk_level="CRITICAL", created_at=None)

    status = EmergencyResponseService.evaluate_step(unavail_camera, risk_assessment=assessment_undated)
    assert status["state"] == "NORMAL"


# --- SECTION 27: PEOPLE COUNT SEMANTICS ---

def test_people_count_none_reports_unavailable():
    """people_count=None -> unavailable, never claims area clear."""
    fire_1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 701}
    fire_2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 702}
    EmergencyResponseService.evaluate_step(fire_1, person_count=None, current_timestamp=100.0)
    status = EmergencyResponseService.evaluate_step(fire_2, person_count=None, current_timestamp=100.5)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["visible_people_count"] is None
    assert status["people_detection_available"] is False
    assert status["priority"] == "CRITICAL"


def test_people_count_zero_reports_unconfirmed_occupancy():
    """people_count=0 -> 0 people currently detected, actual occupancy not confirmed."""
    fire_1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 703}
    fire_2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 704}
    EmergencyResponseService.evaluate_step(fire_1, person_count=0, current_timestamp=100.0)
    status = EmergencyResponseService.evaluate_step(fire_2, person_count=0, current_timestamp=100.5)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["visible_people_count"] == 0
    assert status["people_detection_available"] is True
    assert status["priority"] == "HIGH"


def test_people_count_positive_reports_detected_people():
    """people_count=2 -> reports 2 visible occupants, CRITICAL priority."""
    fire_1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 705}
    fire_2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.70, "inference_id": 706}
    EmergencyResponseService.evaluate_step(fire_1, person_count=2, current_timestamp=100.0)
    status = EmergencyResponseService.evaluate_step(fire_2, person_count=2, current_timestamp=100.5)
    assert status["state"] == "EMERGENCY_CONFIRMED"
    assert status["visible_people_count"] == 2
    assert status["people_detection_available"] is True
    assert status["priority"] == "CRITICAL"


# --- SECTION 28 & 6: EMAIL & SMTP CONFIGURATION TESTS ---

@patch("smtplib.SMTP")
def test_email_disabled_does_not_call_smtp(mock_smtp):
    """When EMAIL_NOTIFICATIONS_ENABLED=False, SMTP must not be called."""
    with patch.object(settings, "EMAIL_NOTIFICATIONS_ENABLED", False):
        res = EmailNotificationService.send_emergency_notification({"state": "EMERGENCY_CONFIRMED"})
        assert res is False
        mock_smtp.assert_not_called()


@patch("smtplib.SMTP")
def test_smtp_auth_required_semantics(mock_smtp):
    """When SMTP_REQUIRE_AUTH=True, missing username or password results in configured=False."""
    with patch.object(settings, "SMTP_HOST", "smtp.gmail.com"), \
         patch.object(settings, "SMTP_PORT", 587), \
         patch.object(settings, "EMERGENCY_EMAIL_FROM", "from@test.com"), \
         patch.object(settings, "EMERGENCY_EMAIL_TO", "to@test.com"), \
         patch.object(settings, "SMTP_REQUIRE_AUTH", True), \
         patch.object(settings, "SMTP_USERNAME", ""), \
         patch.object(settings, "SMTP_PASSWORD", ""):

        assert EmailNotificationService.is_configured() is False

        # Supply credentials
        with patch.object(settings, "SMTP_USERNAME", "user@test.com"), \
             patch.object(settings, "SMTP_PASSWORD", "secret123"):
            assert EmailNotificationService.is_configured() is True


@patch("smtplib.SMTP")
def test_emergency_transition_sends_one_smtp_email(mock_smtp):
    """Emergency transition triggers exactly 1 SMTP send."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    with patch.object(settings, "EMAIL_NOTIFICATIONS_ENABLED", True), \
         patch.object(settings, "SMTP_HOST", "smtp.test.com"), \
         patch.object(settings, "SMTP_PORT", 587), \
         patch.object(settings, "SMTP_REQUIRE_AUTH", True), \
         patch.object(settings, "SMTP_USERNAME", "user@test.com"), \
         patch.object(settings, "SMTP_PASSWORD", "secret123"), \
         patch.object(settings, "EMERGENCY_EMAIL_FROM", "alert@test.com"), \
         patch.object(settings, "EMERGENCY_EMAIL_TO", "admin@test.com"):

        fire_1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 801}
        fire_2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 802}
        EmergencyResponseService.evaluate_step(fire_1, person_count=1, current_timestamp=100.0)
        EmergencyResponseService.evaluate_step(fire_2, person_count=1, current_timestamp=100.5)

        time.sleep(0.1)  # Allow background email dispatch thread to complete
        assert mock_server.send_message.call_count == 1


@patch("smtplib.SMTP")
def test_continuing_emergency_does_not_send_repeated_emails(mock_smtp):
    """Continuing emergency does NOT send duplicate emails (anti-spam episode policy)."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    with patch.object(settings, "EMAIL_NOTIFICATIONS_ENABLED", True), \
         patch.object(settings, "SMTP_HOST", "smtp.test.com"), \
         patch.object(settings, "SMTP_PORT", 587), \
         patch.object(settings, "SMTP_REQUIRE_AUTH", True), \
         patch.object(settings, "SMTP_USERNAME", "user@test.com"), \
         patch.object(settings, "SMTP_PASSWORD", "secret123"), \
         patch.object(settings, "EMERGENCY_EMAIL_FROM", "alert@test.com"), \
         patch.object(settings, "EMERGENCY_EMAIL_TO", "admin@test.com"):

        f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 810}
        f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 811}
        EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
        EmergencyResponseService.evaluate_step(f2, current_timestamp=100.5)
        time.sleep(0.1)
        assert mock_server.send_message.call_count == 1

        # Additional fire frames while emergency is already confirmed
        for i in range(3):
            fn = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 820 + i}
            EmergencyResponseService.evaluate_step(fn, current_timestamp=101.0 + i)

        time.sleep(0.1)
        assert mock_server.send_message.call_count == 1  # Still exactly 1


@patch("smtplib.SMTP")
def test_new_emergency_after_clear_sends_second_email(mock_smtp):
    """After confirmed emergency clears and a NEW emergency occurs, a second email is allowed."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    with patch.object(settings, "EMAIL_NOTIFICATIONS_ENABLED", True), \
         patch.object(settings, "SMTP_HOST", "smtp.test.com"), \
         patch.object(settings, "SMTP_PORT", 587), \
         patch.object(settings, "SMTP_REQUIRE_AUTH", True), \
         patch.object(settings, "SMTP_USERNAME", "user@test.com"), \
         patch.object(settings, "SMTP_PASSWORD", "secret123"), \
         patch.object(settings, "EMERGENCY_EMAIL_FROM", "alert@test.com"), \
         patch.object(settings, "EMERGENCY_EMAIL_TO", "admin@test.com"):

        # Episode 1 confirmation
        f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 901}
        f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 902}
        EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
        EmergencyResponseService.evaluate_step(f2, current_timestamp=100.5)
        time.sleep(0.1)
        assert mock_server.send_message.call_count == 1

        # Clear emergency with 5 fresh zeros
        for i in range(5):
            z = {"detection_available": True, "fire_detected": False, "fire_confidence": 0.0, "inference_id": 910 + i}
            EmergencyResponseService.evaluate_step(z, current_timestamp=102.0 + i)

        assert EmergencyResponseService.get_status()["state"] == "NORMAL"

        # Episode 2 confirmation
        f3 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 920}
        f4 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 921}
        EmergencyResponseService.evaluate_step(f3, current_timestamp=110.0)
        EmergencyResponseService.evaluate_step(f4, current_timestamp=110.5)
        time.sleep(0.1)
        assert mock_server.send_message.call_count == 2


@patch("smtplib.SMTP")
def test_smtp_failure_records_error_and_does_not_crash(mock_smtp):
    """SMTP failure records error truthfully without crashing backend."""
    mock_smtp.side_effect = Exception("Connection refused by SMTP server")

    with patch.object(settings, "EMAIL_NOTIFICATIONS_ENABLED", True), \
         patch.object(settings, "SMTP_HOST", "smtp.test.com"), \
         patch.object(settings, "SMTP_PORT", 587), \
         patch.object(settings, "SMTP_REQUIRE_AUTH", True), \
         patch.object(settings, "SMTP_USERNAME", "user@test.com"), \
         patch.object(settings, "SMTP_PASSWORD", "secret123"), \
         patch.object(settings, "EMERGENCY_EMAIL_FROM", "alert@test.com"), \
         patch.object(settings, "EMERGENCY_EMAIL_TO", "admin@test.com"):

        res = EmailNotificationService.send_emergency_notification({"state": "EMERGENCY_CONFIRMED"})
        assert res is False
        status = EmailNotificationService.get_status()
        assert status["last_error"] is not None
        assert "Connection refused" in status["last_error"]


# --- SECTION 5: REAL .env CONFIGURATION LOADING ---

def test_env_file_configuration():
    """Verify Settings is configured to load backend/.env and respects path resolution."""
    assert os.path.isabs(str(ENV_FILE_PATH))
    assert str(ENV_FILE_PATH).endswith(".env")
    assert settings.model_config.get("case_sensitive") is True


# --- SECTION 7: MONITOR LIFECYCLE TESTS ---

def test_monitor_lifecycle_repeated_start_single_worker():
    """start_monitor called repeatedly must create only one worker."""
    try:
        EmergencyResponseService.stop_monitor()
        EmergencyResponseService.start_monitor()
        thread_1 = EmergencyResponseService._monitor_thread
        assert thread_1 is not None
        assert thread_1.is_alive()

        # Call start_monitor again
        EmergencyResponseService.start_monitor()
        thread_2 = EmergencyResponseService._monitor_thread
        assert thread_1 is thread_2  # Same worker, no duplicate spawned
    finally:
        EmergencyResponseService.stop_monitor()


def test_monitor_lifecycle_stop_and_clean_restart():
    """stop_monitor cleanly exits and clears references; restart spawns exactly one new worker."""
    try:
        EmergencyResponseService.stop_monitor()
        EmergencyResponseService.start_monitor()
        t1 = EmergencyResponseService._monitor_thread
        assert t1 is not None

        EmergencyResponseService.stop_monitor()
        assert EmergencyResponseService._monitor_thread is None
        assert EmergencyResponseService._stop_event is None
        assert not t1.is_alive()

        # Restart
        EmergencyResponseService.start_monitor()
        t2 = EmergencyResponseService._monitor_thread
        assert t2 is not None
        assert t2 is not t1
        assert t2.is_alive()
    finally:
        EmergencyResponseService.stop_monitor()


# --- SECTION 13 & 19: API ENDPOINT TESTS ---

def test_emergency_status_api(client):
    """Verify GET /api/v1/emergency/status endpoint structure."""
    response = client.get("/api/v1/emergency/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "priority" in data
    assert "reason" in data
    assert "confirmation_threshold" in data
    assert data["confirmation_threshold"] == 0.50
    assert "temporal_hits" in data
    assert "people_detection_available" in data
    assert "confirmation_source" in data
    assert "observation_id" in data
    assert "observation_timestamp" in data
    assert "last_updated_at" in data


def test_emergency_email_status_api_hides_secrets(client):
    """Verify GET /api/v1/emergency/email/status endpoint never leaks SMTP password."""
    response = client.get("/api/v1/emergency/email/status")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "configured" in data
    assert "password" not in data
    assert "SMTP_PASSWORD" not in data


# --- SECTION 32: ARCHITECTURAL BOUNDARY AUDIT ---

def test_architectural_invariants():
    """Verify architectural boundaries: no YOLO/VideoCapture inside Phase 5B emergency or email services."""
    import inspect
    import app.services.emergency_response_service as ers_module
    import app.services.email_notification_service as ens_module

    ers_src = inspect.getsource(ers_module)
    ens_src = inspect.getsource(ens_module)

    # EmergencyResponseService must not perform YOLO inference or open VideoCapture
    assert "cv2.VideoCapture" not in ers_src
    assert "VideoCapture(" not in ers_src
    assert "model.predict(" not in ers_src
    assert "YOLO(" not in ers_src

    # EmailNotificationService must not perform YOLO inference or video capture
    assert "cv2.VideoCapture" not in ens_src
    assert "VideoCapture(" not in ens_src
    assert "model.predict(" not in ens_src
    assert "YOLO(" not in ens_src

    # Model path remains unchanged
    assert settings.FIRE_MODEL_PATH == "models/best.pt"


# --- SECTION 33: PRODUCER/CONSUMER INFERENCE-ID TESTS ---

class MockBox:
    def __init__(self, conf: float, cls_id: int, xyxy: list):
        self.conf = [conf]
        self.cls = [cls_id]
        self._xyxy = xyxy

    @property
    def xyxy(self):
        class XYXY:
            def __init__(self, val):
                self.val = val

            def tolist(self):
                return self.val

            def __getitem__(self, idx):
                return self.val[idx]

        return XYXY([self._xyxy])


class MockResult:
    def __init__(self, boxes: list):
        self.boxes = boxes


class MockFireYOLO:
    def __init__(self, names=None, boxes_to_return=None):
        self.names = names if names is not None else {0: "smoke", 1: "fire"}
        self.boxes_to_return = boxes_to_return or []

    def predict(self, source, **kwargs):
        return [MockResult(self.boxes_to_return)]


def test_producer_repeated_calls_same_inference_id():
    """
    Test A: One published successful inference
    -> get_fresh_summary() called repeatedly
    -> same inference_id every time
    """
    try:
        CameraService.stop_camera()
        FireDetectionService.stop_worker()

        boxes = [MockBox(conf=0.88, cls_id=1, xyxy=[10.0, 20.0, 100.0, 200.0])]
        mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=boxes)
        FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with CameraService._lock:
            CameraService._is_connected = True
            CameraService._latest_frame = dummy_frame
            CameraService._generation = 1
            FireDetectionService._camera_session_id = 1

        stop_ev = threading.Event()
        with FireDetectionService._lock:
            FireDetectionService._worker_generation = 1
            FireDetectionService._stop_event = stop_ev

        with patch.object(time, "sleep", side_effect=lambda s: stop_ev.set()):
            FireDetectionService._inference_worker(stop_ev, generation=1)

        s1 = FireDetectionService.get_fresh_summary()
        s2 = FireDetectionService.get_fresh_summary()
        s3 = FireDetectionService.get_fresh_summary()

        assert s1["detection_available"] is True
        assert s1["inference_id"] is not None
        assert s1["inference_id"] == s2["inference_id"] == s3["inference_id"]
        assert s1["inference_timestamp"] == s2["inference_timestamp"] == s3["inference_timestamp"]
    finally:
        CameraService.stop_camera()
        FireDetectionService.stop_worker()
        with FireDetectionService._lock:
            FireDetectionService._model = None
            FireDetectionService._model_status = "Not Loaded"
            FireDetectionService.clear_detections()


def test_producer_next_inference_increments_monotonically():
    """
    Test B: Next genuinely successful published inference
    -> inference_id increments monotonically
    """
    try:
        CameraService.stop_camera()
        FireDetectionService.stop_worker()

        boxes = [MockBox(conf=0.88, cls_id=1, xyxy=[10.0, 20.0, 100.0, 200.0])]
        mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=boxes)
        FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with CameraService._lock:
            CameraService._is_connected = True
            CameraService._latest_frame = dummy_frame
            CameraService._generation = 1
            FireDetectionService._camera_session_id = 1

        stop_ev1 = threading.Event()
        with FireDetectionService._lock:
            FireDetectionService._worker_generation = 1
            FireDetectionService._stop_event = stop_ev1

        with patch.object(time, "sleep", side_effect=lambda s: stop_ev1.set()):
            FireDetectionService._inference_worker(stop_ev1, generation=1)

        s1 = FireDetectionService.get_fresh_summary()
        id1 = s1["inference_id"]
        ts1 = s1["inference_timestamp"]

        stop_ev2 = threading.Event()
        with FireDetectionService._lock:
            FireDetectionService._worker_generation = 1
            FireDetectionService._stop_event = stop_ev2

        with patch.object(time, "sleep", side_effect=lambda s: stop_ev2.set()):
            FireDetectionService._inference_worker(stop_ev2, generation=1)

        s2 = FireDetectionService.get_fresh_summary()
        id2 = s2["inference_id"]
        ts2 = s2["inference_timestamp"]

        assert id2 > id1
        assert id2 == id1 + 1
        assert ts2 >= ts1
    finally:
        CameraService.stop_camera()
        FireDetectionService.stop_worker()
        with FireDetectionService._lock:
            FireDetectionService._model = None
            FireDetectionService._model_status = "Not Loaded"
            FireDetectionService.clear_detections()


def test_consumer_same_inference_identity_counts_exactly_once():
    """
    Test C: EmergencyResponseService receiving the same actual inference identity multiple times
    -> counts exactly once
    """
    EmergencyResponseService.reset_state()
    summary = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.85,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 42,
        "inference_timestamp": 1720000000.0,
    }

    res1 = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.0)
    assert res1["temporal_hits"] == 1
    assert res1["observation_id"] == 42
    assert res1["observation_timestamp"] == 1720000000.0

    res2 = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.4)
    assert res2["temporal_hits"] == 1
    assert res2["observation_id"] == 42

    res3 = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.8)
    assert res3["temporal_hits"] == 1
    assert res3["observation_id"] == 42


def test_consumer_fresh_fire_missing_identity_does_not_increment_hits():
    """
    Test D: Fresh fire summary missing stable inference identity
    -> does NOT increment temporal_hits
    """
    EmergencyResponseService.reset_state()
    summary = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.90,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": None,
        "inference_timestamp": None,
    }

    res = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.0)
    # Fail closed: must not count as a new temporal confirmation hit
    assert res["temporal_hits"] == 0
    # Must NOT confirm emergency despite confidence >= 0.50
    assert res["state"] != "EMERGENCY_CONFIRMED"
    assert res["state"] == "MONITORING"
    # But current semantic metrics are still displayed
    assert res["fire_detection_available"] is True
    assert res["fire_confidence"] == 0.90
    assert res["observation_id"] is None
    assert res["observation_timestamp"] is None

    # Repeated calls still do not increment temporal_hits or confirm emergency
    for i in range(5):
        res2 = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.4 + i * 0.4)
        assert res2["temporal_hits"] == 0
        assert res2["state"] != "EMERGENCY_CONFIRMED"


def test_consumer_fresh_zero_missing_identity_does_not_advance_clearing():
    """
    Test E: Fresh zero summary missing stable inference identity
    -> does NOT advance visual clearing counter
    """
    EmergencyResponseService.reset_state()
    # Confirm an emergency with two distinct valid inferences
    f1 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 1, "inference_timestamp": 100.0}
    f2 = {"detection_available": True, "fire_detected": True, "fire_confidence": 0.85, "inference_id": 2, "inference_timestamp": 101.0}
    EmergencyResponseService.evaluate_step(f1, current_timestamp=100.0)
    confirmed_status = EmergencyResponseService.evaluate_step(f2, current_timestamp=101.0)
    assert confirmed_status["state"] == "EMERGENCY_CONFIRMED"
    assert EmergencyResponseService._consecutive_negative_samples == 0

    # Present 10 fresh zero summaries that lack stable inference identity (inference_id: None)
    zero_missing_id = {
        "detection_available": True,
        "fire_detected": False,
        "fire_confidence": 0.0,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": None,
        "inference_timestamp": None,
    }
    for i in range(10):
        s = EmergencyResponseService.evaluate_step(zero_missing_id, current_timestamp=102.0 + i * 0.4)
        assert s["state"] == "EMERGENCY_CONFIRMED"
        assert EmergencyResponseService._consecutive_negative_samples == 0

    assert EmergencyResponseService._consecutive_negative_samples == 0
    assert EmergencyResponseService.get_status()["state"] == "EMERGENCY_CONFIRMED"


def test_observation_id_and_timestamp_strict_separation():
    """
    Verify observation_id and observation_timestamp semantics:
    - observation_id is an integer sequence ID
    - observation_timestamp is a UNIX timestamp
    - sequence number is NEVER labeled or stored as observation_timestamp
    """
    EmergencyResponseService.reset_state()
    summary = {
        "detection_available": True,
        "fire_detected": True,
        "fire_confidence": 0.85,
        "smoke_detected": False,
        "smoke_confidence": 0.0,
        "inference_id": 7,
        "inference_timestamp": 1728001234.56,
    }
    status = EmergencyResponseService.evaluate_step(summary, current_timestamp=100.0)
    assert status["observation_id"] == 7
    assert isinstance(status["observation_id"], int)
    assert status["observation_timestamp"] == 1728001234.56
    assert isinstance(status["observation_timestamp"], float)
    assert status["observation_timestamp"] != float(status["observation_id"])
