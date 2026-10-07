"""
Phase 4 Automated Test Suite: Real Custom D-Fire YOLO Fire & Smoke Detection.
Uses mocked YOLO and synthetic OpenCV frames to guarantee 100% offline, deterministic,
zero-GPU, zero-webcam execution while verifying all Phase 4 architectural invariants.
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
from app.services.fire_detection_service import FireDetectionService
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


class MockFireYOLO:
    def __init__(self, names=None, boxes_to_return=None):
        self.names = names if names is not None else {0: "smoke", 1: "fire"}
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
    FireDetectionService.stop_worker()

    with PersonDetectionService._lock:
        PersonDetectionService._model = None
        PersonDetectionService._model_status = "Not Loaded"
        PersonDetectionService._person_class_id = None
        PersonDetectionService._latest_detections = []
        PersonDetectionService._latest_person_count = 0
        PersonDetectionService._last_inference_time = 0.0
        PersonDetectionService._camera_session_id = 0

    with FireDetectionService._lock:
        FireDetectionService._model = None
        FireDetectionService._model_status = "Not Loaded"
        FireDetectionService._fire_class_id = None
        FireDetectionService._smoke_class_id = None
        FireDetectionService._latest_detections = []
        FireDetectionService._fire_detected = None
        FireDetectionService._smoke_detected = None
        FireDetectionService._fire_confidence = None
        FireDetectionService._smoke_confidence = None
        FireDetectionService._last_inference_time = 0.0
        FireDetectionService._camera_session_id = 0
        FireDetectionService._last_inference_error = None

    yield

    CameraService.stop_camera()
    PersonDetectionService.stop_worker()
    FireDetectionService.stop_worker()

    with FireDetectionService._lock:
        FireDetectionService._model = None
        FireDetectionService._model_status = "Not Loaded"
        FireDetectionService._fire_class_id = None
        FireDetectionService._smoke_class_id = None
        FireDetectionService._latest_detections = []
        FireDetectionService._fire_detected = None
        FireDetectionService._smoke_detected = None
        FireDetectionService._fire_confidence = None
        FireDetectionService._smoke_confidence = None
        FireDetectionService._last_inference_time = 0.0
        FireDetectionService._camera_session_id = 0
        FireDetectionService._last_inference_error = None


@pytest.fixture
def client():
    return TestClient(app)


# 1. Successful Dynamic Class Resolution
def test_successful_dynamic_class_resolution():
    # Case 1: Standard mapping {0: 'smoke', 1: 'fire'}
    mock_model = MockFireYOLO(names={0: "smoke", 1: "fire"})
    with patch("ultralytics.YOLO", return_value=mock_model):
        loaded = FireDetectionService.initialize_model("dummy_path.pt")
        assert loaded is True
        assert FireDetectionService.is_model_loaded() is True
        assert FireDetectionService._smoke_class_id == 0
        assert FireDetectionService._fire_class_id == 1

    # Case 2: Inverted mapping {0: 'fire', 1: 'smoke'}
    mock_model_inv = MockFireYOLO(names={0: "fire", 1: "smoke"})
    with patch("ultralytics.YOLO", return_value=mock_model_inv):
        loaded = FireDetectionService.initialize_model("dummy_path.pt")
        assert loaded is True
        assert FireDetectionService._smoke_class_id == 1
        assert FireDetectionService._fire_class_id == 0

    # Case 3: Case-insensitive discovery
    mock_model_case = MockFireYOLO(names={10: "SMOKE", 20: "FiRe"})
    with patch("ultralytics.YOLO", return_value=mock_model_case):
        loaded = FireDetectionService.initialize_model("dummy_path.pt")
        assert loaded is True
        assert FireDetectionService._smoke_class_id == 10
        assert FireDetectionService._fire_class_id == 20


# 2. Failure When Fire Class Missing
def test_failure_when_fire_class_missing():
    mock_model = MockFireYOLO(names={0: "smoke", 1: "person", 2: "car"})
    with patch("ultralytics.YOLO", return_value=mock_model):
        loaded = FireDetectionService.initialize_model("dummy_path.pt")
        assert loaded is False
        assert FireDetectionService.is_model_loaded() is False
        assert "Required class(es) 'fire' not found" in FireDetectionService.get_model_status()
        assert FireDetectionService._fire_class_id is None
        assert FireDetectionService._smoke_class_id is None


# 3. Failure When Smoke Class Missing
def test_failure_when_smoke_class_missing():
    mock_model = MockFireYOLO(names={0: "fire", 1: "person"})
    with patch("ultralytics.YOLO", return_value=mock_model):
        loaded = FireDetectionService.initialize_model("dummy_path.pt")
        assert loaded is False
        assert FireDetectionService.is_model_loaded() is False
        assert "Required class(es) 'smoke' not found" in FireDetectionService.get_model_status()
        assert FireDetectionService._fire_class_id is None
        assert FireDetectionService._smoke_class_id is None


# 4. Correct Fire Detection Parsing
def test_correct_fire_detection_parsing():
    boxes = [
        MockBox(conf=0.62, cls_id=1, xyxy=[10.0, 20.0, 80.0, 90.0]),
        MockBox(conf=0.87, cls_id=1, xyxy=[15.0, 25.0, 85.0, 95.0]),
    ]
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

    summary = FireDetectionService.get_fresh_summary()
    assert summary["detection_available"] is True
    assert summary["fire_detected"] is True
    assert summary["fire_confidence"] == 0.87
    assert summary["smoke_detected"] is False
    assert summary["smoke_confidence"] == 0.0

    dets = FireDetectionService.get_fresh_detections()
    assert len(dets) == 2
    assert all(d["class_name"] == "fire" for d in dets)


# 5. Correct Smoke Detection Parsing
def test_correct_smoke_detection_parsing():
    boxes = [
        MockBox(conf=0.74, cls_id=0, xyxy=[5.0, 10.0, 150.0, 200.0]),
        MockBox(conf=0.55, cls_id=0, xyxy=[20.0, 30.0, 100.0, 150.0]),
    ]
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

    summary = FireDetectionService.get_fresh_summary()
    assert summary["detection_available"] is True
    assert summary["smoke_detected"] is True
    assert summary["smoke_confidence"] == 0.74
    assert summary["fire_detected"] is False
    assert summary["fire_confidence"] == 0.0


# 6. Fire and Smoke Simultaneously
def test_fire_and_smoke_simultaneously():
    boxes = [
        MockBox(conf=0.92, cls_id=1, xyxy=[50.0, 50.0, 150.0, 150.0]),  # Fire
        MockBox(conf=0.81, cls_id=0, xyxy=[10.0, 10.0, 300.0, 300.0]),  # Smoke
    ]
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

    summary = FireDetectionService.get_fresh_summary()
    assert summary["detection_available"] is True
    assert summary["fire_detected"] is True
    assert summary["fire_confidence"] == 0.92
    assert summary["smoke_detected"] is True
    assert summary["smoke_confidence"] == 0.81


# 7. Successful Inference with Zero Detections
def test_successful_inference_with_zero_detections():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
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

    summary = FireDetectionService.get_fresh_summary()
    assert summary["detection_available"] is True
    assert summary["fire_detected"] is False
    assert summary["smoke_detected"] is False
    assert summary["fire_confidence"] == 0.0
    assert summary["smoke_confidence"] == 0.0
    assert FireDetectionService.get_fresh_detections() == []


# 8. Zero Detection vs Unavailable Semantics
def test_zero_detection_vs_unavailable_semantics():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    # State A: Valid inference with 0 detections
    with FireDetectionService._lock:
        FireDetectionService._latest_detections = []
        FireDetectionService._fire_detected = False
        FireDetectionService._smoke_detected = False
        FireDetectionService._fire_confidence = 0.0
        FireDetectionService._smoke_confidence = 0.0
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        status_a = FireDetectionService.get_status()
        assert status_a["detection_available"] is True
        assert status_a["fire_detected"] is False
        assert status_a["smoke_detected"] is False
        assert status_a["fire_confidence"] == 0.0
        assert status_a["smoke_confidence"] == 0.0

    # State B: Camera stopped / detector unavailable
    FireDetectionService.clear_detections()
    with patch.object(CameraService, "is_connected", return_value=False):
        status_b = FireDetectionService.get_status()
        assert status_b["detection_available"] is False
        assert status_b["fire_detected"] is None
        assert status_b["smoke_detected"] is None
        assert status_b["fire_confidence"] is None
        assert status_b["smoke_confidence"] is None


# 9. Stale Result Invalidation
def test_stale_result_invalidation():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.88
        FireDetectionService._smoke_detected = False
        FireDetectionService._smoke_confidence = 0.0
        # Age is older than stale timeout (2.0s)
        FireDetectionService._last_inference_time = time.time() - 3.5
        FireDetectionService._camera_session_id = 1

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        assert FireDetectionService.is_detection_fresh() is False
        summary = FireDetectionService.get_fresh_summary()
        assert summary["detection_available"] is False
        assert summary["fire_detected"] is None

        latest = FireDetectionService.get_latest()
        assert latest["stale"] is True
        assert latest["detection_available"] is False


# 10. Camera Generation Mismatch Rejection
def test_camera_generation_mismatch_rejection():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.95
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1  # Session 1

    # Camera is now on Session 2
    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=2):
        assert FireDetectionService.is_detection_fresh() is False
        latest = FireDetectionService.get_latest()
        assert latest["detection_available"] is False
        assert latest["stale"] is True
        assert latest["fire_detected"] is None


# 11. Disconnected Camera Produces Unavailable Result
def test_disconnected_camera_produces_unavailable_result():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = False
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1

    with patch.object(CameraService, "is_connected", return_value=False):
        assert FireDetectionService.is_detection_fresh() is False
        summary = FireDetectionService.get_fresh_summary()
        assert summary["detection_available"] is False
        assert summary["fire_detected"] is None


# 12. Model Failure Produces Unavailable Result
def test_model_failure_produces_unavailable_result():
    with patch("ultralytics.YOLO", side_effect=RuntimeError("Corrupt weights file")):
        loaded = FireDetectionService.initialize_model("corrupt.pt")
        assert loaded is False
        assert FireDetectionService.is_model_loaded() is False
        assert "Error: Corrupt weights file" in FireDetectionService.get_model_status()

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        assert FireDetectionService.is_detection_fresh() is False
        status = FireDetectionService.get_status()
        assert status["model_loaded"] is False
        assert status["detection_available"] is False
        assert status["fire_detected"] is None


# 13. Inference Exception Invalidates Freshness Immediately
def test_inference_exception_invalidates_freshness_immediately():
    mock_faulty_yolo = MagicMock()
    mock_faulty_yolo.predict.side_effect = RuntimeError("CUDA Out of Memory")
    mock_faulty_yolo.names = {0: "smoke", 1: "fire"}
    FireDetectionService.set_mock_model(mock_faulty_yolo, fire_class_id=1, smoke_class_id=0)

    # Start with seemingly fresh detection
    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.9
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    stop_ev = threading.Event()
    with FireDetectionService._lock:
        FireDetectionService._worker_generation = 1
        FireDetectionService._stop_event = stop_ev

    with patch.object(time, "sleep", side_effect=lambda s: stop_ev.set()):
        FireDetectionService._inference_worker(stop_ev, generation=1)

    # Freshness must be wiped out immediately
    assert FireDetectionService._last_inference_time == 0.0
    assert FireDetectionService._latest_detections == []
    assert FireDetectionService._last_inference_error == "CUDA Out of Memory"
    assert FireDetectionService.is_detection_fresh() is False


# 14. Duplicate Worker Start Protection
def test_duplicate_worker_start_protection():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with patch.object(CameraService, "is_connected", return_value=False):
        started1 = FireDetectionService.start_worker()
        assert started1 is True
        w1 = FireDetectionService._worker_thread
        assert w1 is not None and w1.is_alive()

        # Attempt to start worker again -> must return True and reuse existing singleton
        started2 = FireDetectionService.start_worker()
        assert started2 is True
        assert FireDetectionService._worker_thread is w1

        FireDetectionService.stop_worker()
        assert FireDetectionService._worker_thread is None


# 15. Worker Stop/Start Lifecycle
def test_worker_stop_start_lifecycle():
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with patch.object(CameraService, "is_connected", return_value=False):
        assert FireDetectionService.start_worker() is True
        w1 = FireDetectionService._worker_thread
        assert w1 is not None and w1.is_alive()

        FireDetectionService.stop_worker()
        assert not w1.is_alive()
        assert FireDetectionService._worker_thread is None

        # Restart
        assert FireDetectionService.start_worker() is True
        w2 = FireDetectionService._worker_thread
        assert w2 is not None and w2.is_alive()
        assert w2 is not w1

        FireDetectionService.stop_worker()


# 16. No FireDetectionService VideoCapture Ownership
def test_no_fire_detection_service_videocapture_ownership():
    import inspect
    import app.services.fire_detection_service as fds_module

    source_code = inspect.getsource(fds_module)
    assert "cv2.VideoCapture(" not in source_code
    assert "VideoCapture(" not in source_code
    assert 'source=0' not in source_code
    assert 'source="0"' not in source_code


# 17. /fire/status API
def test_fire_status_api(client: TestClient):
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.89
        FireDetectionService._smoke_detected = False
        FireDetectionService._smoke_confidence = 0.0
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1
        FireDetectionService._last_inference_latency_ms = 45.2
        FireDetectionService._inference_fps = 22.1

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        res = client.get("/api/v1/detection/fire/status")
        assert res.status_code == 200
        data = res.json()
        assert data["model_loaded"] is True
        assert data["status"] == "Loaded"
        assert data["detection_available"] is True
        assert data["fire_detected"] is True
        assert data["fire_confidence"] == 0.89
        assert data["smoke_detected"] is False
        assert data["smoke_confidence"] == 0.0
        assert data["inference_fps"] == 22.1
        assert data["inference_latency_ms"] == 45.2


# 18. /fire/latest API
def test_fire_latest_api(client: TestClient):
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    with FireDetectionService._lock:
        FireDetectionService._latest_detections = [
            {"bbox": [10.0, 20.0, 100.0, 150.0], "confidence": 0.84, "class_name": "fire"}
        ]
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.84
        FireDetectionService._smoke_detected = False
        FireDetectionService._smoke_confidence = 0.0
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        res = client.get("/api/v1/detection/fire/latest")
        assert res.status_code == 200
        data = res.json()
        assert data["model_loaded"] is True
        assert data["camera_connected"] is True
        assert data["detection_available"] is True
        assert data["stale"] is False
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_name"] == "fire"
        assert data["fire_detected"] is True
        assert data["fire_confidence"] == 0.84


# 19. Camera Start Safely Activates Person + Fire Services
def test_camera_start_safely_activates_person_and_fire_services():
    p_mock = MagicMock()
    p_mock.names = {0: "person"}
    PersonDetectionService.set_mock_model(p_mock, person_class_id=0)

    f_mock = MockFireYOLO(names={0: "smoke", 1: "fire"})
    FireDetectionService.set_mock_model(f_mock, fire_class_id=1, smoke_class_id=0)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        res = CameraService.start_camera(0)
        assert res["connected"] is True

        assert PersonDetectionService._worker_thread is not None
        assert PersonDetectionService._worker_thread.is_alive()

        assert FireDetectionService._worker_thread is not None
        assert FireDetectionService._worker_thread.is_alive()

        CameraService.stop_camera()
        PersonDetectionService.stop_worker()
        FireDetectionService.stop_worker()


# 20. Camera Stop Clears Both Services
def test_camera_stop_clears_both_services():
    with PersonDetectionService._lock:
        PersonDetectionService._latest_person_count = 2
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._camera_session_id = 1

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.91
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1

    CameraService.stop_camera()

    assert PersonDetectionService._last_inference_time == 0.0
    assert PersonDetectionService._latest_person_count == 0
    assert PersonDetectionService.get_fresh_person_count() is None

    assert FireDetectionService._last_inference_time == 0.0
    assert FireDetectionService._latest_detections == []
    assert FireDetectionService.get_fresh_summary()["detection_available"] is False


# 21. Permanent Camera Failure Notifies Both Services
def test_permanent_camera_failure_notifies_both():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 1 success read then all fails
    mock_cap.read.side_effect = [(True, dummy_frame)] + [(False, None)] * 40
    mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 30.0)

    with PersonDetectionService._lock:
        PersonDetectionService._latest_person_count = 1
        PersonDetectionService._last_inference_time = time.time()

    with FireDetectionService._lock:
        FireDetectionService._fire_detected = True
        FireDetectionService._fire_confidence = 0.85
        FireDetectionService._last_inference_time = time.time()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        CameraService.start_camera(0)
        time.sleep(0.6)  # Allow worker to hit > 30 failures and trigger hooks

        assert CameraService.is_connected() is False
        assert PersonDetectionService.get_fresh_person_count() is None
        assert FireDetectionService.get_fresh_summary()["detection_available"] is False


# 22. Existing Phase 1-3 Invariants Still Pass
def test_existing_phase1_3_invariants_pass():
    # Phase 1 weights must sum to 1.0 and remain unchanged
    w_sum = (
        settings.WEIGHT_TEMPERATURE +
        settings.WEIGHT_SMOKE +
        settings.WEIGHT_GAS +
        settings.WEIGHT_FLAME +
        settings.WEIGHT_CAMERA_FIRE
    )
    assert round(w_sum, 2) == 1.0
    assert settings.FIRE_CONFIDENCE_THRESHOLD == 0.30
    assert settings.FIRE_CONFIRMATION_THRESHOLD == 0.50


# 23. Person Detection Semantics Remain Unchanged
def test_person_detection_semantics_remain_unchanged():
    p_mock = MagicMock()
    p_mock.names = {0: "person"}
    PersonDetectionService.set_mock_model(p_mock, person_class_id=0)

    with PersonDetectionService._lock:
        PersonDetectionService._latest_person_count = 3
        PersonDetectionService._latest_detections = [
            {"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.88, "class_name": "person"}
        ]
        PersonDetectionService._last_inference_time = time.time()
        PersonDetectionService._camera_session_id = 1

    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):
        assert PersonDetectionService.get_fresh_person_count() == 3
        status = PersonDetectionService.get_status()
        assert status["people_count"] == 3
        assert status["detection_available"] is True

        # Frame annotation preserves raw frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated = PersonDetectionService.annotate_frame(test_frame.copy())
        assert annotated is not test_frame


# 24. Risk Calculation Remains Unchanged (Phase 5 Isolation)
def test_risk_calculation_remains_unchanged():
    reading = SensorReading(
        temperature=25.0,
        humidity=50.0,
        smoke_level=5.0,
        gas_level=5.0,
        flame_level=0.0,
        input_source="manual_simulation"
    )
    db_mock = MagicMock()
    assessment, alert = RiskAssessmentService.evaluate_and_record(
        reading=reading,
        db=db_mock,
        camera_fire_conf=None,  # Phase 5 reserved
        people_count=None
    )
    assert assessment.camera_fire_risk is None
    assert assessment.risk_level == "SAFE"


# 25. Concurrency & Deadlock Prevention Test
def test_lock_order_concurrency_no_deadlock():
    """
    Deterministic concurrency test verifying that concurrent execution of
    FireDetectionService.on_camera_started() and FireDetectionService.stop_worker()
    across threads completes with zero deadlocks.
    """
    mock_yolo = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[])
    FireDetectionService.set_mock_model(mock_yolo, fire_class_id=1, smoke_class_id=0)

    for _ in range(25):
        errors = []

        def worker_start():
            try:
                FireDetectionService.on_camera_started()
            except Exception as e:
                errors.append(e)

        def worker_stop():
            try:
                FireDetectionService.stop_worker(timeout=0.5)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_start)
        t2 = threading.Thread(target=worker_stop)

        t1.start()
        t2.start()

        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert not t1.is_alive(), "Worker start thread deadlocked!"
        assert not t2.is_alive(), "Worker stop thread deadlocked!"
        assert len(errors) == 0, f"Encountered unexpected concurrency errors: {errors}"

    FireDetectionService.stop_worker()


# 26. Regression Test: Old Fire Worker Must Never Publish After Stop
def test_old_fire_worker_never_publishes_after_stop():
    predict_started = threading.Event()
    predict_release = threading.Event()

    class BlockingMockModel:
        names = {0: "smoke", 1: "fire"}

        def predict(self, source, **kwargs):
            predict_started.set()
            predict_release.wait(timeout=3.0)
            box = MockBox(conf=0.95, cls_id=1, xyxy=[10.0, 10.0, 50.0, 50.0])
            return [MockResult([box])]

    blocking_model = BlockingMockModel()
    FireDetectionService.set_mock_model(blocking_model, fire_class_id=1, smoke_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    # Start the real worker thread
    started = FireDetectionService.start_worker()
    assert started is True
    w_thread = FireDetectionService._worker_thread
    assert w_thread is not None and w_thread.is_alive()

    # Wait until predict() has entered
    assert predict_started.wait(timeout=2.0) is True, "Worker did not enter predict()!"

    # Call stop_worker with tiny timeout so worker remains blocked inside predict()
    FireDetectionService.stop_worker(timeout=0.01)

    # Worker must still be alive inside predict()
    assert w_thread.is_alive(), "Worker finished before predict was unblocked!"

    # Confirm detections were cleared by stop_worker()
    assert FireDetectionService._latest_detections == []
    assert FireDetectionService._last_inference_time == 0.0
    assert FireDetectionService.get_fresh_summary()["detection_available"] is False

    # Unblock predict()
    predict_release.set()

    # Wait for the old worker thread to terminate
    w_thread.join(timeout=2.0)
    assert not w_thread.is_alive(), "Worker failed to terminate!"

    # CRITICAL INVARIANT: The old inference result must NOT have been published!
    assert FireDetectionService._last_inference_time == 0.0
    assert FireDetectionService.get_fresh_summary()["detection_available"] is False
    assert FireDetectionService._latest_detections == []
    assert FireDetectionService._fire_detected is None


# 27. Regression Test: Obsolete Worker Generation Cannot Publish
def test_obsolete_worker_generation_cannot_publish():
    box = MockBox(conf=0.88, cls_id=1, xyxy=[20.0, 20.0, 60.0, 60.0])
    mock_model = MockFireYOLO(names={0: "smoke", 1: "fire"}, boxes_to_return=[box])
    FireDetectionService.set_mock_model(mock_model, fire_class_id=1, smoke_class_id=0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CameraService._lock:
        CameraService._is_connected = True
        CameraService._latest_frame = dummy_frame
        CameraService._generation = 1

    stop_ev = threading.Event()

    # Run inference worker with obsolete generation = 1 when current _worker_generation = 2
    with FireDetectionService._lock:
        FireDetectionService._worker_generation = 2
        FireDetectionService._stop_event = threading.Event()  # different stop event

    with patch.object(time, "sleep", side_effect=lambda s: stop_ev.set()):
        FireDetectionService._inference_worker(stop_ev, generation=1)

    # Obsolete generation must have its result rejected and cleared
    assert FireDetectionService._last_inference_time == 0.0
    assert FireDetectionService._latest_detections == []
    assert FireDetectionService.get_fresh_summary()["detection_available"] is False


# 28. Regression Test: SystemService Follows Real FireDetectionService State (Single Source of Truth)
def test_system_service_tracks_fire_detection_service_truth():
    db_mock = MagicMock()

    # State 1: Genuine Not Loaded
    with FireDetectionService._lock:
        FireDetectionService._model = None
        FireDetectionService._model_status = "Not Loaded"

    status_1 = SystemService.get_status(db_mock)
    assert status_1.fire_model == "Fire AI Model Not Loaded"

    # State 2: Loaded
    mock_model = MockFireYOLO(names={0: "smoke", 1: "fire"})
    FireDetectionService.set_mock_model(mock_model, fire_class_id=1, smoke_class_id=0)
    status_2 = SystemService.get_status(db_mock)
    assert status_2.fire_model == "Model Loaded & Initialized"

    # State 3: Real Model Loading Error
    with FireDetectionService._lock:
        FireDetectionService._model = None
        FireDetectionService._model_status = "Error: Corrupted model file"

    status_3 = SystemService.get_status(db_mock)
    assert status_3.fire_model == "Error: Corrupted model file"

    # Reset
    with FireDetectionService._lock:
        FireDetectionService._model = None
        FireDetectionService._model_status = "Not Loaded"


# 29. Presentation Overlay Preserves Raw Frame
def test_fire_presentation_overlay_preserves_raw_frame():
    with FireDetectionService._lock:
        FireDetectionService._latest_detections = [
            {"bbox": [10.0, 10.0, 100.0, 100.0], "confidence": 0.88, "class_name": "fire"},
            {"bbox": [120.0, 10.0, 200.0, 100.0], "confidence": 0.75, "class_name": "smoke"},
        ]
        FireDetectionService._fire_detected = True
        FireDetectionService._smoke_detected = True
        FireDetectionService._fire_confidence = 0.88
        FireDetectionService._smoke_confidence = 0.75
        FireDetectionService._last_inference_time = time.time()
        FireDetectionService._camera_session_id = 1
        FireDetectionService._model_status = "Loaded"
        FireDetectionService._model = MagicMock()

    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(CameraService, "is_connected", return_value=True), \
         patch.object(CameraService, "get_generation", return_value=1):

        copy_for_stream = raw_frame.copy()
        annotated = FireDetectionService.annotate_frame(copy_for_stream)

        # Raw frame was not mutated
        assert np.all(raw_frame == 0)
        # Annotated copy has pixels drawn (bounding boxes and labels)
        assert not np.all(annotated == 0)
        assert annotated is not raw_frame

