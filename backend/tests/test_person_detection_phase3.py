"""
Phase 3 Automated Test Suite: Real YOLO Person Detection & Visible Person Counting.
Uses mocked YOLO and camera frames to ensure 100% offline, deterministic,
zero-GPU, zero-webcam execution while verifying all Phase 3 architectural invariants.
Guarantees database isolation (does not mutate database/fire_system.db).
"""

import time
import threading
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.camera_service import CameraService
from app.services.person_detection_service import PersonDetectionService
from app.services.system_service import SystemService
from app.services.risk_engine import RiskAssessmentService
from app.models.sensor import SensorReading


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


class MockYOLO:
    def __init__(self, names=None, boxes_to_return=None):
        self.names = names if names is not None else {0: "person", 1: "bicycle", 2: "car"}
        self.boxes_to_return = boxes_to_return or []
        self.predict_calls = []

    def predict(self, source, **kwargs):
        self.predict_calls.append({"source": source, "kwargs": kwargs})
        return [MockResult(self.boxes_to_return)]


@pytest.fixture(autouse=True)
def reset_services():
    """Ensure clean state before and after each test."""
    CameraService.stop_camera()
    PersonDetectionService.stop_worker()
    with PersonDetectionService._lock:
        PersonDetectionService._model = None
        PersonDetectionService._model_status = "Not Loaded"
        PersonDetectionService._person_class_id = None
        PersonDetectionService._latest_detections = []
        PersonDetectionService._latest_person_count = 0
        PersonDetectionService._last_inference_time = 0.0
        PersonDetectionService._camera_session_id = 0
    yield
    CameraService.stop_camera()
    PersonDetectionService.stop_worker()


@pytest.fixture
def client():
    return TestClient(app)


# 1. Model Initialization Tests
def test_successful_person_model_initialization():
    mock_yolo = MockYOLO(names={0: "person", 1: "dog"})
    with patch("ultralytics.YOLO", return_value=mock_yolo):
        loaded = PersonDetectionService.initialize_model("dummy_path.pt")
        assert loaded is True
        assert PersonDetectionService.is_model_loaded() is True
        assert PersonDetectionService.get_model_status() == "Loaded"
        assert PersonDetectionService._person_class_id == 0


def test_model_initialization_failure():
    with patch("ultralytics.YOLO", side_effect=RuntimeError("Corrupt weights")):
        loaded = PersonDetectionService.initialize_model("corrupt.pt")
        assert loaded is False
        assert PersonDetectionService.is_model_loaded() is False
        status = PersonDetectionService.get_model_status()
        assert status.startswith("Error: Corrupt weights")


def test_dynamic_person_class_id_discovery():
    # 'person' class is at ID 4 instead of 0
    mock_yolo = MockYOLO(names={0: "chair", 1: "table", 4: "Person", 7: "lamp"})
    with patch("ultralytics.YOLO", return_value=mock_yolo):
        loaded = PersonDetectionService.initialize_model("custom_classes.pt")
        assert loaded is True
        assert PersonDetectionService._person_class_id == 4


def test_missing_person_class_fails_transparently():
    # No 'person' class exists in model class mapping
    mock_yolo = MockYOLO(names={0: "car", 1: "truck", 2: "bus"})
    with patch("ultralytics.YOLO", return_value=mock_yolo):
        loaded = PersonDetectionService.initialize_model("vehicles_only.pt")
        assert loaded is False
        assert PersonDetectionService.is_model_loaded() is False
        assert "not found" in PersonDetectionService.get_model_status()


# 2. Person Filtering & Confidence Threshold Tests
def test_confidence_threshold_and_class_filtering():
    """Verify person >= threshold is kept, low conf person rejected, and other classes rejected."""
    box_person_high = MockBox(conf=0.88, cls_id=0, xyxy=[10.0, 20.0, 100.0, 200.0])
    box_person_low = MockBox(conf=0.35, cls_id=0, xyxy=[15.0, 25.0, 80.0, 150.0])  # < 0.50 rejected
    box_car = MockBox(conf=0.95, cls_id=2, xyxy=[50.0, 50.0, 300.0, 300.0])          # Non-person rejected

    mock_yolo = MockYOLO(
        names={0: "person", 1: "bicycle", 2: "car"},
        boxes_to_return=[box_person_high, box_person_low, box_car]
    )
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    # Simulate camera running with a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1
        PersonDetectionService._camera_session_id = 1

    stop_event = threading.Event()
    # Run one single iteration of worker logic manually or start briefly
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    detections = PersonDetectionService.get_fresh_detections()
    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.88
    assert detections[0]["class_name"] == "person"
    assert detections[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert PersonDetectionService.get_fresh_person_count() == 1


def test_zero_people_detected():
    mock_yolo = MockYOLO(boxes_to_return=[])
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    stop_event = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    assert PersonDetectionService.get_fresh_person_count() == 0
    assert PersonDetectionService.get_fresh_detections() == []


def test_multiple_people_detected():
    box1 = MockBox(conf=0.92, cls_id=0, xyxy=[10.0, 10.0, 50.0, 100.0])
    box2 = MockBox(conf=0.81, cls_id=0, xyxy=[60.0, 10.0, 110.0, 100.0])
    box3 = MockBox(conf=0.74, cls_id=0, xyxy=[120.0, 10.0, 180.0, 100.0])

    mock_yolo = MockYOLO(boxes_to_return=[box1, box2, box3])
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    stop_event = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    assert PersonDetectionService.get_fresh_person_count() == 3
    assert len(PersonDetectionService.get_fresh_detections()) == 3


# 3. Lifecycle, Stale Protection, and Camera Isolation
def test_camera_disconnected_clears_person_detection():
    # Setup fresh detections
    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "person"}]
        PersonDetectionService._latest_person_count = 1
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    # Camera is NOT connected
    with CameraService._lock:
        CameraService._is_connected = False
        CameraService._generation = 1

    assert PersonDetectionService.is_detection_fresh() is False
    assert PersonDetectionService.get_fresh_person_count() is None
    assert PersonDetectionService.get_fresh_detections() == []


def test_stale_detection_protection():
    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "person"}]
        PersonDetectionService._latest_person_count = 1
        # Set inference time older than PERSON_STALE_TIMEOUT (2.0s)
        PersonDetectionService._last_inference_time = time.time() - (settings.PERSON_STALE_TIMEOUT + 5.0)
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    assert PersonDetectionService.is_detection_fresh() is False
    assert PersonDetectionService.get_fresh_person_count() is None
    assert PersonDetectionService.get_fresh_detections() == []


def test_camera_session_restart_clears_old_session_boxes():
    # Detections recorded in generation 1
    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "person"}]
        PersonDetectionService._latest_person_count = 1
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    # Camera advances to generation 2
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 2

    assert PersonDetectionService.is_detection_fresh() is False
    assert PersonDetectionService.get_fresh_person_count() is None
    assert PersonDetectionService.get_fresh_detections() == []


def test_detector_worker_does_not_duplicate():
    mock_yolo = MockYOLO()
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    # Start worker 3 times
    res1 = PersonDetectionService.start_worker()
    t1 = PersonDetectionService._worker_thread
    res2 = PersonDetectionService.start_worker()
    t2 = PersonDetectionService._worker_thread
    res3 = PersonDetectionService.start_worker()
    t3 = PersonDetectionService._worker_thread

    assert res1 is True
    assert res2 is True
    assert res3 is True
    assert t1 is t2 and t2 is t3
    assert t1.is_alive()

    PersonDetectionService.stop_worker()
    assert PersonDetectionService._worker_thread is None


def test_model_failure_does_not_stop_webcam():
    # Set camera connected
    with CameraService._lock:
        CameraService._is_connected = True

    # Person model fails
    with patch("ultralytics.YOLO", side_effect=Exception("GPU out of memory")):
        PersonDetectionService.initialize_model("heavy_model.pt")

    assert PersonDetectionService.is_model_loaded() is False
    # Camera must remain connected!
    assert CameraService.is_connected() is True


def test_presentation_overlay_preserves_raw_frame():
    """Ensure annotate_frame does NOT mutate the original raw frame stored by CameraService."""
    raw_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = [{
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "confidence": 0.95,
            "class_name": "person"
        }]
        PersonDetectionService._latest_person_count = 1
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1
        CameraService._latest_frame = raw_frame

    # Stream layer takes a copy and annotates it
    frame_copy = raw_frame.copy()
    annotated = PersonDetectionService.annotate_frame(frame_copy)

    # The copy should have colored pixels (green bounding box)
    assert np.any(annotated > 0)
    # The raw frame stored in CameraService must remain completely untouched (all zeros)
    assert np.all(CameraService._latest_frame == 0)


# 4. API & System Status Verification
def test_system_status_reports_person_model_state(client):
    # Case A: Not Loaded
    with PersonDetectionService._lock:
        PersonDetectionService._model = None
        PersonDetectionService._model_status = "Not Loaded"

    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    assert res.json()["person_model"] == "Not Loaded"
    assert res.json()["fire_model"] == "Fire AI Model Not Loaded"

    # Case B: Loaded
    with PersonDetectionService._lock:
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._model_status = "Loaded"

    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    assert res.json()["person_model"] == "Loaded"
    assert res.json()["fire_model"] == "Fire AI Model Not Loaded"

    # Case C: Error
    with PersonDetectionService._lock:
        PersonDetectionService._model = None
        PersonDetectionService._model_status = "Error: Driver incompatibility"

    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    assert res.json()["person_model"] == "Error: Driver incompatibility"
    assert res.json()["fire_model"] == "Fire AI Model Not Loaded"


def test_person_detection_status_api(client):
    with PersonDetectionService._lock:
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model_name = "yolo26n.pt"
        PersonDetectionService._latest_person_count = 2
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._camera_session_id = 1
        PersonDetectionService._inference_fps = 8.5
        PersonDetectionService._last_inference_latency_ms = 42.0

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    res = client.get("/api/v1/detection/person/status")
    assert res.status_code == 200
    data = res.json()
    assert data["model_loaded"] is True
    assert data["status"] == "Loaded"
    assert data["model_name"] == "yolo26n.pt"
    assert data["people_count"] == 2
    assert data["detection_available"] is True
    assert data["confidence_threshold"] == 0.50
    assert data["inference_fps"] == 8.5
    assert data["inference_latency_ms"] == 42.0


def test_person_detection_latest_api(client):
    with PersonDetectionService._lock:
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._latest_detections = [{
            "bbox": [50.0, 60.0, 150.0, 300.0],
            "confidence": 0.89,
            "class_name": "person"
        }]
        PersonDetectionService._latest_person_count = 1
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._camera_session_id = 1
        PersonDetectionService._last_inference_latency_ms = 35.0

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    res = client.get("/api/v1/detection/person/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["model_loaded"] is True
    assert data["camera_connected"] is True
    assert data["people_count"] == 1
    assert data["detection_available"] is True
    assert len(data["detections"]) == 1
    assert data["detections"][0]["confidence"] == 0.89
    assert data["stale"] is False


# 5. Deterministic Inference Exception & Immediate Freshness Invalidation (Section 4)
def test_inference_exception_invalidates_freshness_immediately(client):
    """
    Verify: successful inference with person count > 0
    -> immediately run a failing inference
    -> before PERSON_STALE_TIMEOUT expires
    -> get_fresh_person_count() must return None / unavailable
    -> API must not return a valid zero-person state.
    """
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    # Step 1: Successful inference with 2 people
    box1 = MockBox(conf=0.90, cls_id=0, xyxy=[10.0, 10.0, 50.0, 100.0])
    box2 = MockBox(conf=0.85, cls_id=0, xyxy=[60.0, 10.0, 110.0, 100.0])
    mock_yolo_ok = MockYOLO(boxes_to_return=[box1, box2])
    PersonDetectionService.set_mock_model(mock_yolo_ok, person_class_id=0)

    stop_event = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    assert PersonDetectionService.is_detection_fresh() is True
    assert PersonDetectionService.get_fresh_person_count() == 2

    # Step 2: Failing inference occurs
    class FailingModel:
        names = {0: "person"}
        def predict(self, source, **kwargs):
            raise RuntimeError("CUDA or Torch inference crashed!")

    PersonDetectionService.set_mock_model(FailingModel(), person_class_id=0)

    stop_event2 = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event2.set()):
        PersonDetectionService._inference_worker(stop_event2)

    # Step 3: Immediately verify freshness is invalidated BEFORE stale timeout expires
    assert PersonDetectionService._last_inference_time == 0.0
    assert PersonDetectionService.is_detection_fresh() is False
    assert PersonDetectionService.get_fresh_person_count() is None  # MUST NOT BE 0!

    # Step 4: Verify API returns null people_count, detection_available=false, stale=true
    res_latest = client.get("/api/v1/detection/person/latest")
    assert res_latest.status_code == 200
    latest_data = res_latest.json()
    assert latest_data["people_count"] is None
    assert latest_data["detection_available"] is False
    assert latest_data["stale"] is True

    res_status = client.get("/api/v1/detection/person/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["people_count"] is None
    assert status_data["detection_available"] is False


# 6. Deterministic Worker Stop/Restart Race Condition (Section 5 & 6)
def test_deterministic_worker_race_condition():
    """
    Deterministic test with a controlled blocking model:
    predict():
    - signals predict_started
    - blocks on unblock_predict
    Sequence:
    1. Start PersonInferenceWorker.
    2. Confirm it is blocked inside predict().
    3. Call stop_worker() from another controlled thread.
    4. Before old predict is released, attempt start_worker().
    5. Verify no second PersonInferenceWorker is created.
    6. Release the blocked predict.
    7. Verify old worker exits cleanly.
    8. Verify a later start creates exactly one replacement worker.
    9. Verify only one PersonInferenceWorker exists.
    """
    predict_started = threading.Event()
    unblock_predict = threading.Event()

    class ControlledBlockingModel:
        names = {0: "person"}
        def predict(self, source, **kwargs):
            predict_started.set()
            unblock_predict.wait(timeout=5.0)
            return []

    blocking_model = ControlledBlockingModel()
    PersonDetectionService.set_mock_model(blocking_model, person_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    # 1. Start PersonInferenceWorker
    started = PersonDetectionService.start_worker()
    assert started is True
    initial_worker = PersonDetectionService._worker_thread
    assert initial_worker is not None and initial_worker.is_alive()

    # 2. Confirm it is blocked inside predict()
    assert predict_started.wait(timeout=3.0) is True

    # 3. Call stop_worker() from another controlled thread
    stop_called = threading.Event()
    def async_stop():
        # Will attempt to join, but predict is blocked so join times out (use short timeout)
        PersonDetectionService.stop_worker(timeout=0.1)
        stop_called.set()

    stop_thread = threading.Thread(target=async_stop)
    stop_thread.start()
    stop_called.wait(timeout=1.0)
    stop_thread.join(timeout=1.0)

    # Worker must STILL be alive because predict has not unblocked
    assert initial_worker.is_alive() is True
    # Worker reference MUST NOT be cleared while it is still alive
    assert PersonDetectionService._worker_thread is initial_worker

    # 4. Before old predict is released, attempt start_worker()
    second_start_result = PersonDetectionService.start_worker(join_timeout=0.1)

    # 5. Verify no second PersonInferenceWorker is created!
    assert second_start_result is False
    assert PersonDetectionService._worker_thread is initial_worker

    # 6. Release the blocked predict
    unblock_predict.set()

    # 7. Verify old worker exits cleanly
    initial_worker.join(timeout=3.0)
    assert initial_worker.is_alive() is False

    # 8. Verify a later start creates exactly one replacement worker
    restart_result = PersonDetectionService.start_worker()
    assert restart_result is True

    # 9. Verify only one PersonInferenceWorker exists
    replacement_worker = PersonDetectionService._worker_thread
    assert replacement_worker is not None
    assert replacement_worker is not initial_worker
    assert replacement_worker.is_alive()

    # Clean up replacement worker
    PersonDetectionService.stop_worker()
    assert PersonDetectionService._worker_thread is None


# 7. End-to-End Availability Tests (Section 7)
def test_fresh_valid_zero_people_is_available(client):
    """Fresh valid 0: people_count = 0, detection_available = True, stale = False."""
    mock_yolo = MockYOLO(boxes_to_return=[])
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    stop_event = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    assert PersonDetectionService.get_fresh_person_count() == 0
    assert PersonDetectionService.is_detection_fresh() is True

    res = client.get("/api/v1/detection/person/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["people_count"] == 0
    assert data["detection_available"] is True
    assert data["stale"] is False

    res_status = client.get("/api/v1/detection/person/status")
    assert res_status.status_code == 200
    sdata = res_status.json()
    assert sdata["people_count"] == 0
    assert sdata["detection_available"] is True


def test_camera_disconnected_is_unavailable_not_zero(client):
    """When camera is disconnected, people_count must be null and detection_available false."""
    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = []
        PersonDetectionService._latest_person_count = 0
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    with CameraService._lock:
        CameraService._is_connected = False
        CameraService._generation = 1

    assert PersonDetectionService.get_fresh_person_count() is None

    res = client.get("/api/v1/detection/person/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["people_count"] is None
    assert data["detection_available"] is False
    assert data["stale"] is True


def test_model_unavailable_returns_unavailable_not_zero(client):
    """When model is not loaded, people_count must be null and detection_available false."""
    with PersonDetectionService._lock:
        PersonDetectionService._model = None
        PersonDetectionService._model_status = "Not Loaded"
        PersonDetectionService._latest_person_count = 0
        PersonDetectionService._last_inference_time = 0.0

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    assert PersonDetectionService.get_fresh_person_count() is None

    res = client.get("/api/v1/detection/person/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["people_count"] is None
    assert data["detection_available"] is False
    assert data["stale"] is True


def test_stale_detection_returns_unavailable_not_zero(client):
    """When detection is stale (>2.0s), people_count must be null and detection_available false."""
    with PersonDetectionService._lock:
        PersonDetectionService._latest_detections = []
        PersonDetectionService._latest_person_count = 0
        PersonDetectionService._last_inference_time = time.time() - 10.0
        PersonDetectionService._model_status = "Loaded"
        PersonDetectionService._model = MagicMock()
        PersonDetectionService._camera_session_id = 1

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    assert PersonDetectionService.get_fresh_person_count() is None

    res = client.get("/api/v1/detection/person/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["people_count"] is None
    assert data["detection_available"] is False
    assert data["stale"] is True


def test_risk_assessment_emergency_priority_with_unknown_occupancy(db_session):
    """
    CRITICAL requirement:
    - CRITICAL hazard + people_count=None -> CRITICAL priority (does NOT downgrade to HIGH).
      Wording explicitly indicates person-detection status is unavailable.
    - CRITICAL hazard + people_count=0 -> HIGH priority with 0-people wording.
    - CRITICAL hazard + people_count>0 -> CRITICAL priority with visible-person wording.
    - HIGH RISK hazard + people_count=None -> HIGH priority (does NOT downgrade to MEDIUM).
    - HIGH RISK hazard + people_count=0 -> MEDIUM priority.
    - Alert wording NEVER claims '0 people' or 'monitored area clear' when unavailable.
    """
    from datetime import datetime, timezone
    critical_reading = SensorReading(
        temperature=75.0,
        humidity=20.0,
        smoke_level=85.0,
        gas_level=40.0,
        flame_level=90.0,
        input_source="manual_simulation",
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(critical_reading)
    db_session.commit()
    db_session.refresh(critical_reading)

    # 1. CRITICAL hazard with unavailable person detection (people_count=None)
    assessment_none, alert_none = RiskAssessmentService.evaluate_and_record(
        reading=critical_reading,
        db=db_session,
        people_count=None
    )
    assert assessment_none.emergency_priority == "CRITICAL"
    assert assessment_none.people_detection_available is False
    assert alert_none is not None
    assert alert_none.people_detection_available is False
    assert "Person-detection status is currently unavailable; occupancy could not be determined" in alert_none.message
    assert "0 people" not in alert_none.message
    assert "clear" not in alert_none.message.lower()

    # 2. CRITICAL hazard with confirmed zero people (people_count=0)
    assessment_zero, alert_zero = RiskAssessmentService.evaluate_and_record(
        reading=critical_reading,
        db=db_session,
        people_count=0
    )
    assert assessment_zero.emergency_priority == "HIGH"
    assert assessment_zero.people_detection_available is True
    assert assessment_zero.people_detected == 0
    assert "0 people currently detected" in alert_zero.message

    # 3. CRITICAL hazard with confirmed people (people_count=3)
    assessment_three, alert_three = RiskAssessmentService.evaluate_and_record(
        reading=critical_reading,
        db=db_session,
        people_count=3
    )
    assert assessment_three.emergency_priority == "CRITICAL"
    assert assessment_three.people_detection_available is True
    assert assessment_three.people_detected == 3
    assert "3 people currently detected" in alert_three.message

    # 4. HIGH RISK hazard with unavailable person detection
    high_reading = SensorReading(
        temperature=65.0,
        humidity=30.0,
        smoke_level=68.0,
        gas_level=55.0,
        flame_level=75.0,
        input_source="manual_simulation",
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(high_reading)
    db_session.commit()
    db_session.refresh(high_reading)

    assessment_high_none, alert_high_none = RiskAssessmentService.evaluate_and_record(
        reading=high_reading,
        db=db_session,
        people_count=None
    )
    assert assessment_high_none.emergency_priority == "HIGH"
    assert assessment_high_none.people_detection_available is False
    assert alert_high_none is not None
    assert "Person-detection status is currently unavailable; occupancy could not be determined" in alert_high_none.message
    assert "0 people" not in alert_high_none.message
    assert "clear" not in alert_high_none.message.lower()

    # 5. HIGH RISK hazard with confirmed zero people
    assessment_high_zero, alert_high_zero = RiskAssessmentService.evaluate_and_record(
        reading=high_reading,
        db=db_session,
        people_count=0
    )
    assert assessment_high_zero.emergency_priority == "MEDIUM"
    assert assessment_high_zero.people_detection_available is True
    assert "0 people currently detected by camera" in alert_high_zero.message
    assert "Actual occupancy is not confirmed" in alert_high_zero.message


def test_sensor_submission_endpoint_preserves_unavailable_vs_zero(client):
    """Verify POST /api/v1/sensors/readings passes unavailable as None and returns null people_detected."""
    payload = {
        "temperature": 75.0,
        "humidity": 20.0,
        "smoke_level": 85.0,
        "gas_level": 30.0,
        "flame_level": 90.0,
        "input_source": "manual_simulation"
    }

    # Case A: Person detection unavailable
    with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=None):
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["risk"]["people_detection_available"] is False
        assert data["risk"]["people_detected"] is None
        assert data["risk"]["emergency_priority"] == "CRITICAL"
        if data["active_alert"]:
            assert data["active_alert"]["people_detection_available"] is False
            assert data["active_alert"]["people_detected"] is None
            assert "unavailable" in data["active_alert"]["message"]
            assert "0 people" not in data["active_alert"]["message"]

    # Case B: Person detection fresh valid 0
    with patch.object(PersonDetectionService, "get_fresh_person_count", return_value=0):
        res0 = client.post("/api/v1/sensors/readings", json=payload)
        assert res0.status_code == 201
        data0 = res0.json()
        assert data0["risk"]["people_detection_available"] is True
        assert data0["risk"]["people_detected"] == 0
        assert data0["risk"]["emergency_priority"] == "HIGH"
        if data0["active_alert"]:
            assert data0["active_alert"]["people_detection_available"] is True
            assert data0["active_alert"]["people_detected"] == 0
            assert "0 people" in data0["active_alert"]["message"]


# 8. Concurrency & Lock-Order Integrity Tests
def test_lock_order_concurrency_no_deadlock():
    """
    Deterministic test verifying that concurrent on_camera_started() and stop_worker()
    execute cleanly without deadlock.
    Ensures strict global lock hierarchy (_lifecycle_lock -> _lock, never _lock -> _lifecycle_lock).
    """
    mock_yolo = MockYOLO()
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._generation = 1

    errors = []

    for i in range(25):
        def run_camera_started():
            try:
                PersonDetectionService.on_camera_started()
            except Exception as e:
                errors.append(e)

        def run_stop_worker():
            try:
                PersonDetectionService.stop_worker(timeout=1.0)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_camera_started, name=f"Tester-CamStart-{i}")
        t2 = threading.Thread(target=run_stop_worker, name=f"Tester-StopWorker-{i}")

        t1.start()
        t2.start()

        # Both must complete well within the timeout; if lock inversion existed, deadlock would occur
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert not t1.is_alive(), f"Deadlock detected on iteration {i}: run_camera_started thread hung!"
        assert not t2.is_alive(), f"Deadlock detected on iteration {i}: run_stop_worker thread hung!"

    assert len(errors) == 0, f"Concurrent execution produced errors: {errors}"
    PersonDetectionService.stop_worker()


def test_model_state_participates_in_detection_freshness(client):
    """
    Verify: fresh successful result exists
    -> force model initialization/state failure
    -> timestamp would otherwise still be fresh
    -> get_fresh_person_count() = None
    -> status/latest people_count = null
    -> detection_available = false
    -> stale = true for latest endpoint
    API must NEVER return: model_loaded=false and detection_available=true.
    """
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    box1 = MockBox(conf=0.90, cls_id=0, xyxy=[10.0, 10.0, 50.0, 100.0])
    mock_yolo = MockYOLO(boxes_to_return=[box1])
    PersonDetectionService.set_mock_model(mock_yolo, person_class_id=0)

    stop_event = threading.Event()
    with patch.object(time, "sleep", side_effect=lambda s: stop_event.set()):
        PersonDetectionService._inference_worker(stop_event)

    # Detections are currently fresh and available
    assert PersonDetectionService.is_detection_fresh() is True
    assert PersonDetectionService.get_fresh_person_count() == 1

    # Force model initialization / state failure
    with patch("ultralytics.YOLO", side_effect=RuntimeError("Corrupt model file")):
        init_ok = PersonDetectionService.initialize_model("nonexistent_corrupt.pt")
        assert init_ok is False
        assert PersonDetectionService.is_model_loaded() is False
        assert "Error" in PersonDetectionService.get_model_status()

    # Even though camera is connected and time elapsed is < 2.0s:
    # 1. Freshness must be invalidated because model is not loaded
    assert PersonDetectionService.is_detection_fresh() is False
    assert PersonDetectionService.get_fresh_person_count() is None

    # 2. Latest endpoint must return people_count: null, detection_available: false, stale: true
    res_latest = client.get("/api/v1/detection/person/latest")
    assert res_latest.status_code == 200
    latest_data = res_latest.json()
    assert latest_data["model_loaded"] is False
    assert latest_data["people_count"] is None
    assert latest_data["detection_available"] is False
    assert latest_data["stale"] is True

    # 3. Status endpoint must return people_count: null, detection_available: false
    res_status = client.get("/api/v1/detection/person/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["model_loaded"] is False
    assert status_data["people_count"] is None
    assert status_data["detection_available"] is False
