"""
Phase 2 Automated Unit Tests for Webcam Integration and OpenCV Video Streaming.
All tests use mocked cv2.VideoCapture to run deterministically in CI/CD and
hardware-free environments without requiring a physical camera connected.
"""

import time
import threading
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


from app.main import app
from app.services.camera_service import CameraService
from tests.conftest import TestingSessionLocal

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_camera_state():
    """Ensure camera is stopped before and after every test."""
    CameraService.stop_camera()
    yield
    CameraService.stop_camera()


def create_dummy_cv2_frame(width=640, height=480):
    """Generate a valid dummy BGR frame for testing."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_camera_start_success_mocked():
    """Verify camera starts successfully with mock hardware and updates status."""
    dummy_frame = create_dummy_cv2_frame()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # 1. Start camera
        start_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert start_res.status_code == 200, f"Expected 200, got {start_res.text}"
        data = start_res.json()
        assert data["connected"] is True
        assert data["status"] == "Connected"
        assert data["camera_index"] == 0
        assert data["width"] == 640
        assert data["height"] == 480
        assert data["fps"] == 30.0
        assert data["error"] is None

        # 2. Verify status endpoint reflects Connected state
        status_res = client.get("/api/v1/camera/status")
        assert status_res.status_code == 200
        assert status_res.json()["connected"] is True
        assert status_res.json()["status"] == "Connected"

        # 3. Stop camera
        stop_res = client.post("/api/v1/camera/stop")
        assert stop_res.status_code == 200
        assert stop_res.json()["connected"] is False
        assert stop_res.json()["status"] == "Disconnected"


def test_camera_start_hardware_unavailable_mocked():
    """Verify camera returns 503 and transparent error when hardware fails to open."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        res = client.post("/api/v1/camera/start", json={"camera_index": 9})
        assert res.status_code == 503
        assert "unable to open camera" in res.json()["detail"].lower()

        # Status must report Disconnected or Error, never Connected
        status_res = client.get("/api/v1/camera/status")
        assert status_res.json()["connected"] is False
        assert "Connected" != status_res.json()["status"]


def test_camera_start_initial_frame_read_failure():
    """Verify that if camera opens but fails to read initial frame, it cleanly releases and reports error."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert res.status_code == 503
        assert "failed to capture initial frame" in res.json()["detail"].lower()
        mock_cap.release.assert_called()
        assert CameraService.is_connected() is False


def test_camera_repeated_start_does_not_create_duplicate_instances():
    """Verify repeated start on same index reuses active capture without spawning extra instances."""
    dummy_frame = create_dummy_cv2_frame()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc_class:
        # First start
        res1 = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert res1.status_code == 200
        assert mock_vc_class.call_count == 1

        # Second start on same index
        res2 = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert res2.status_code == 200
        # Must not re-instantiate VideoCapture
        assert mock_vc_class.call_count == 1

        CameraService.stop_camera()


def test_camera_repeated_stop_does_not_crash():
    """Verify calling stop multiple times is idempotent and never raises errors."""
    res1 = client.post("/api/v1/camera/stop")
    assert res1.status_code == 200
    assert res1.json()["connected"] is False

    res2 = client.post("/api/v1/camera/stop")
    assert res2.status_code == 200
    assert res2.json()["connected"] is False

    res3 = client.post("/api/v1/camera/stop")
    assert res3.status_code == 200
    assert res3.json()["status"] == "Disconnected"


def test_camera_stream_endpoint_disconnected_returns_503():
    """Verify accessing stream when camera is stopped returns 503."""
    assert CameraService.is_connected() is False
    res = client.get("/api/v1/camera/stream")
    assert res.status_code == 503
    assert "webcam is disconnected" in res.json()["detail"].lower()


def test_camera_stream_endpoint_connected():
    """Verify streaming response content-type and generator yield when camera is connected."""
    dummy_frame = create_dummy_cv2_frame()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        start_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert start_res.status_code == 200

        # Verify generator directly yields multipart JPEG frame
        gen = CameraService.generate_frames()
        chunk = next(gen)
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk

        CameraService.stop_camera()



def test_system_status_reflects_camera_service():
    """
    Verify GET /api/v1/system/status reflects actual CameraService state.
    Strictly verifies:
    - Camera: Connected / Disconnected
    - Person Model: Not Loaded
    - Fire Model: Fire AI Model Not Loaded
    - Sensor Mode: Manual Simulation
    """
    dummy_frame = create_dummy_cv2_frame()

    # When disconnected
    status_off = client.get("/api/v1/system/status").json()
    assert status_off["camera"] == "Disconnected"
    assert status_off["person_model"] == "Not Loaded"
    assert status_off["fire_model"] == "Fire AI Model Not Loaded"
    assert status_off["sensor_mode"] == "Manual Simulation"

    # When connected
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        client.post("/api/v1/camera/start", json={"camera_index": 0})
        status_on = client.get("/api/v1/system/status").json()
        assert status_on["camera"] == "Connected"
        # Must NOT claim models are loaded
        assert status_on["person_model"] == "Not Loaded"
        assert status_on["fire_model"] == "Fire AI Model Not Loaded"
        assert status_on["sensor_mode"] == "Manual Simulation"

        CameraService.stop_camera()

        # After stop
        status_after = client.get("/api/v1/system/status").json()
        assert status_after["camera"] == "Disconnected"


def test_get_frame_method_future_architecture():
    """Verify CameraService.get_frame() returns frame copy when connected, None when stopped."""
    dummy_frame = create_dummy_cv2_frame(width=320, height=240)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 320.0 if prop == 3 else (240.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        CameraService.start_camera(camera_index=0)
        frame = CameraService.get_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)

        CameraService.stop_camera()
        assert CameraService.get_frame() is None


def test_camera_start_with_custom_index():
    """Verify camera can be requested with a specific hardware index."""
    dummy_frame = create_dummy_cv2_frame()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
        res = client.post("/api/v1/camera/start", json={"camera_index": 2})
        assert res.status_code == 200
        assert res.json()["camera_index"] == 2
        mock_vc.assert_called_with(2)

        CameraService.stop_camera()


def test_manual_sensor_simulation_compatible_with_camera_running():
    """
    Verify Phase 1 compatibility: Manual sensor submissions and risk fusion
    continue operating normally while the webcam is actively streaming.
    """
    dummy_frame = create_dummy_cv2_frame()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # 1. Start camera
        start_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert start_res.status_code == 200
        assert CameraService.is_connected() is True

        # 2. Submit sensor reading while camera is running
        payload = {
            "temperature": 55.0,
            "humidity": 40.0,
            "smoke_level": 65.0,
            "gas_level": 40.0,
            "flame_level": 60.0,
            "input_source": "manual_simulation",
            "scenario_tag": "POSSIBLE_FIRE"
        }
        res = client.post("/api/v1/sensors/readings", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["reading"]["temperature"] == 55.0
        assert data["risk"]["risk_level"] in ["WARNING", "HIGH RISK"]
        assert data["active_alert"] is not None

        # 3. Stop camera and verify reading still persists in database
        CameraService.stop_camera()
        latest = client.get("/api/v1/sensors/latest").json()
        assert latest["temperature"] == 55.0


def test_permanent_frame_read_failure_releases_camera():
    """
    1. PERMANENT FRAME-READ FAILURE MUST RELEASE THE CAMERA:
    - Starts camera with mocked VideoCapture (initial read succeeds).
    - Subsequent reads fail (>30 times).
    - Worker triggers permanent failure handling.
    - Verifies:
        * connected == False
        * VideoCapture.release() was called
        * error status is retained
        * restarting camera creates a fresh capture successfully without calling stop()
    """
    dummy_frame = create_dummy_cv2_frame()
    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = True
    # Initial read succeeds for startup verification, then subsequent reads fail
    mock_cap1.read.side_effect = [(True, dummy_frame)] + [(False, None)] * 40
    mock_cap1.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap1):
        start_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert start_res.status_code == 200
        assert CameraService.is_connected() is True

        # Wait for worker thread to encounter >30 failures and trigger automatic cleanup
        timeout = 2.0
        start_wait = time.time()
        while CameraService.is_connected() and (time.time() - start_wait < timeout):
            time.sleep(0.05)

        # Assertions
        assert CameraService.is_connected() is False, "Camera must transition to disconnected"
        mock_cap1.release.assert_called()
        status_info = CameraService.get_camera_info()
        assert status_info["connected"] is False
        assert "Error:" in status_info["status"]
        assert status_info["error"] is not None
        assert "capture failed irrecoverably" in status_info["error"].lower()

    # Verify restarting the camera creates a fresh capture successfully without requiring an explicit stop call
    mock_cap2 = MagicMock()
    mock_cap2.isOpened.return_value = True
    mock_cap2.read.return_value = (True, dummy_frame)
    mock_cap2.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap2):
        restart_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert restart_res.status_code == 200
        assert restart_res.json()["connected"] is True
        assert restart_res.json()["status"] == "Connected"
        assert CameraService.is_connected() is True

        CameraService.stop_camera()


def test_camera_index_switch_worker_race():
    """
    2. CAMERA INDEX SWITCH WORKER RACE:
    - Camera 0 running -> switch to camera 1.
    - Verify:
        * camera 0 is released
        * old worker exits
        * exactly one worker is active
        * only new worker consumes camera 1
        * service reports camera_index = 1
    """
    dummy_frame = create_dummy_cv2_frame()

    mock_cap0 = MagicMock(name="Cap0")
    mock_cap0.isOpened.return_value = True
    mock_cap0.read.return_value = (True, dummy_frame)
    mock_cap0.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    mock_cap1 = MagicMock(name="Cap1")
    mock_cap1.isOpened.return_value = True
    mock_cap1.read.return_value = (True, dummy_frame)
    mock_cap1.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    def cap_factory(idx):
        return mock_cap0 if idx == 0 else mock_cap1

    with patch("cv2.VideoCapture", side_effect=cap_factory):
        # 1. Start camera 0
        res0 = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert res0.status_code == 200
        assert res0.json()["camera_index"] == 0

        # Verify initial worker thread running
        worker_threads_before = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name]
        assert len(worker_threads_before) == 1
        worker0 = worker_threads_before[0]

        # 2. Switch to camera 1
        res1 = client.post("/api/v1/camera/start", json={"camera_index": 1})
        assert res1.status_code == 200
        assert res1.json()["camera_index"] == 1

        # Verify camera 0 was released
        mock_cap0.release.assert_called()

        # Verify old worker has terminated
        worker0.join(timeout=1.0)
        assert not worker0.is_alive(), "Previous worker thread must have exited"

        # Verify exactly one active worker thread exists now
        worker_threads_after = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name and t.is_alive()]
        assert len(worker_threads_after) == 1, f"Expected 1 active worker, found {len(worker_threads_after)}"

        # Verify service reports camera_index = 1
        info = CameraService.get_camera_info()
        assert info["camera_index"] == 1
        assert info["connected"] is True

        CameraService.stop_camera()


def test_rapid_camera_index_switching():
    """
    Test rapid 0 -> 1 -> 0 switching:
    Verify there is still only one active worker and no thread leak.
    """
    dummy_frame = create_dummy_cv2_frame()

    def make_mock_cap(idx):
        m = MagicMock(name=f"Cap_{idx}")
        m.isOpened.return_value = True
        m.read.return_value = (True, dummy_frame)
        m.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)
        return m

    caps = {}
    def cap_factory(idx):
        caps[idx] = make_mock_cap(idx)
        return caps[idx]

    with patch("cv2.VideoCapture", side_effect=cap_factory):
        # 0 -> 1 -> 0
        client.post("/api/v1/camera/start", json={"camera_index": 0})
        client.post("/api/v1/camera/start", json={"camera_index": 1})
        client.post("/api/v1/camera/start", json={"camera_index": 0})

        time.sleep(0.1)
        active_workers = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name and t.is_alive()]
        assert len(active_workers) == 1, f"Expected exactly 1 active worker, found {len(active_workers)}"

        info = CameraService.get_camera_info()
        assert info["camera_index"] == 0
        assert info["connected"] is True

        CameraService.stop_camera()
        time.sleep(0.1)
        active_workers_final = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name and t.is_alive()]
        assert len(active_workers_final) == 0


def test_failure_retry_lifecycle():
    """
    3. FAILURE -> RETRY LIFECYCLE:
    Connected -> repeated read failure -> automatic release/error state -> Start Camera -> Connected again.
    Ensure no stale capture object or stale worker remains.
    """
    dummy_frame = create_dummy_cv2_frame()

    # Session 1: Fails after initial frame
    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = True
    mock_cap1.read.side_effect = [(True, dummy_frame)] + [(False, None)] * 40
    mock_cap1.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap1):
        client.post("/api/v1/camera/start", json={"camera_index": 0})
        # Wait for worker failure handling
        start_wait = time.time()
        while CameraService.is_connected() and (time.time() - start_wait < 2.0):
            time.sleep(0.05)

        assert CameraService.is_connected() is False
        mock_cap1.release.assert_called()

    # Session 2: Fresh Start Camera
    mock_cap2 = MagicMock()
    mock_cap2.isOpened.return_value = True
    mock_cap2.read.return_value = (True, dummy_frame)
    mock_cap2.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap2):
        res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert res.status_code == 200
        assert res.json()["connected"] is True
        assert CameraService.is_connected() is True

        CameraService.stop_camera()


def test_stop_clears_error_state():
    """
    4. STOP/ERROR STATE CONSISTENCY:
    - Automatic hardware failure transitions to 'Error: <msg>'
    - Explicit Stop transitions to 'Disconnected' and clears stale error message.
    """
    dummy_frame = create_dummy_cv2_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, dummy_frame)] + [(False, None)] * 40
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        client.post("/api/v1/camera/start", json={"camera_index": 0})
        start_wait = time.time()
        while CameraService.is_connected() and (time.time() - start_wait < 2.0):
            time.sleep(0.05)

        # In error state
        info_err = CameraService.get_camera_info()
        assert info_err["connected"] is False
        assert "Error:" in info_err["status"]

        # Explicit Stop
        res_stop = client.post("/api/v1/camera/stop")
        assert res_stop.status_code == 200
        data_stop = res_stop.json()
        assert data_stop["connected"] is False
        assert data_stop["status"] == "Disconnected"
        assert data_stop["error"] is None


def test_negative_camera_index_rejected_validation():
    """
    5. OPTIONAL CAMERA INDEX VALIDATION:
    Verify negative integers return HTTP 422 validation response.
    """
    res = client.post("/api/v1/camera/start", json={"camera_index": -1})
    assert res.status_code == 422
    assert "greater than or equal to 0" in res.text or "non-negative" in res.text


def test_deterministic_failure_worker_release_race():
    """
    DETERMINISTIC RACE TEST:
    Verifies that when a capture worker encounters permanent failure (>30 failed reads):
    1. cap.release() is called, but deliberately paused.
    2. While cap.release() is paused, start_camera() is called concurrently.
    3. start_camera() MUST wait and NOT open a replacement VideoCapture while the old worker
       is still releasing its capture handle.
    4. When old release is unblocked, it completes, the old worker exits cleanly,
       and the replacement VideoCapture is opened.
    5. The service transitions to Connected, exactly one worker exists, and no stale worker/capture remains.
    """
    dummy_frame = create_dummy_cv2_frame()

    release_started = threading.Event()
    release_blocker = threading.Event()
    old_release_finished = threading.Event()
    replacement_vc_opened = threading.Event()

    # Mock 1: Failing Camera
    mock_cap1 = MagicMock(name="FailingCap1")
    mock_cap1.isOpened.return_value = True
    mock_cap1.read.side_effect = [(True, dummy_frame)] + [(False, None)] * 40
    mock_cap1.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    def mock1_release():
        release_started.set()
        # Deliberately pause release until test unblocks it
        release_blocker.wait(timeout=5.0)
        old_release_finished.set()

    mock_cap1.release.side_effect = mock1_release

    # Mock 2: Replacement Camera
    mock_cap2 = MagicMock(name="ReplacementCap2")
    mock_cap2.isOpened.return_value = True
    mock_cap2.read.return_value = (True, dummy_frame)
    mock_cap2.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

    def vc_factory(idx):
        if not release_started.is_set():
            return mock_cap1
        else:
            # Critical assertion: Old release MUST be finished before replacement VideoCapture is instantiated
            if not old_release_finished.is_set():
                raise RuntimeError("Race violation: Replacement VideoCapture opened before previous worker release completed!")
            replacement_vc_opened.set()
            return mock_cap2

    with patch("cv2.VideoCapture", side_effect=vc_factory):
        # 1. Start camera initial session
        start_res = client.post("/api/v1/camera/start", json={"camera_index": 0})
        assert start_res.status_code == 200
        assert CameraService.is_connected() is True

        # 2. Wait until worker encounters >30 failures and begins releasing
        assert release_started.wait(timeout=3.0), "Worker failed to trigger release after >30 read failures"

        # At this point, the old worker is actively paused inside cap1.release()
        assert not old_release_finished.is_set(), "Old release should be blocked"
        assert not replacement_vc_opened.is_set(), "Replacement should not be opened yet"

        # 3. Call start_camera() from another thread while old worker's release() is blocked
        restart_result = {}
        def restart_worker():
            res = client.post("/api/v1/camera/start", json={"camera_index": 0})
            restart_result["status_code"] = res.status_code
            restart_result["json"] = res.json()

        restart_thread = threading.Thread(target=restart_worker, name="ControlledRestartCaller")
        restart_thread.start()

        # 4. Wait a short interval to verify restart_thread is blocked and does NOT open replacement
        time.sleep(0.2)
        assert restart_thread.is_alive(), "start_camera() must block while old worker is still releasing"
        assert not replacement_vc_opened.is_set(), "Replacement VideoCapture must NOT be opened while release is blocked"
        assert not old_release_finished.is_set(), "Old release must still be paused"

        # 5. Now unblock old release
        release_blocker.set()

        # 6. Wait for restart_thread to complete
        restart_thread.join(timeout=3.0)
        assert not restart_thread.is_alive(), "start_camera() should have completed after release unblocked"
        assert restart_result.get("status_code") == 200, f"Expected 200, got {restart_result}"

        # 7. Verify post-conditions
        assert old_release_finished.is_set(), "Old release must have finished"
        assert replacement_vc_opened.is_set(), "Replacement VideoCapture must have been opened"
        assert CameraService.is_connected() is True, "CameraService must be connected"
        info = CameraService.get_camera_info()
        assert info["connected"] is True
        assert info["status"] == "Connected"
        assert info["error"] is None

        # Verify exactly one active capture worker exists
        active_workers = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name and t.is_alive()]
        assert len(active_workers) == 1, f"Expected exactly 1 active worker, found {len(active_workers)}"

        # 8. Clean stop
        stop_res = client.post("/api/v1/camera/stop")
        assert stop_res.status_code == 200
        time.sleep(0.1)
        active_workers_after = [t for t in threading.enumerate() if "CameraCaptureWorker" in t.name and t.is_alive()]
        assert len(active_workers_after) == 0, f"Expected 0 active workers after stop, found {len(active_workers_after)}"



