"""
Camera service for Phase 2: Real OpenCV webcam capture and video streaming.
Manages a single controlled VideoCapture hardware instance with a background
capture worker thread, providing non-blocking frame retrieval for the MJPEG stream
and future Phase 3 (Person Detection) and Phase 4 (Fire Detection) modules.
Includes robust instance ownership, generation tracking, and permanent failure release.
"""

import time
import threading
from typing import Optional, Dict, Any, Generator
import cv2
import numpy as np

from app.core.config import settings


class CameraService:
    """
    Thread-safe OpenCV Camera Manager with explicit worker ownership and generation tracking.
    Controls hardware acquisition, background capture, and frame distribution.
    Guarantees that when the camera is not running or hardware is inaccessible,
    errors are accurately surfaced without fabricating frames.
    """
    _lifecycle_lock: threading.RLock = threading.RLock()
    _lock: threading.RLock = threading.RLock()
    _frame_condition: threading.Condition = threading.Condition(_lock)
    _worker_stop_event: Optional[threading.Event] = None
    _generation: int = 0
    _capture: Optional[cv2.VideoCapture] = None
    _thread: Optional[threading.Thread] = None

    _is_connected: bool = False
    _camera_index: int = settings.CAMERA_INDEX
    _error_message: Optional[str] = None
    _actual_width: int = 0
    _actual_height: int = 0
    _actual_fps: float = 0.0
    _latest_frame: Optional[np.ndarray] = None
    _latest_frame_time: float = 0.0
    _consecutive_failures: int = 0

    @classmethod
    def start_camera(cls, camera_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Open physical webcam using OpenCV VideoCapture.
        Starts a dedicated background capture thread to ensure non-blocking frame retrieval.
        Does not fabricate frames if hardware is unavailable.
        Guarantees that any previous capture worker is safely stopped and joined before
        starting a new capture instance, preventing dual-worker race conditions.
        """
        target_index = camera_index if camera_index is not None else settings.CAMERA_INDEX
        if target_index < 0:
            with cls._lock:
                cls._error_message = "Camera index must be a non-negative integer (>= 0)."
                return cls._get_info_locked()

        with cls._lifecycle_lock:
            # Step 1: If an existing camera is running, check if it's the exact same camera
            thread_to_join = None
            with cls._lock:
                if cls._is_connected and cls._capture is not None and cls._capture.isOpened():
                    if target_index == cls._camera_index:
                        return cls._get_info_locked()
                # If switching camera index or restarting from stale/failed state, cleanly stop old worker
                thread_to_join = cls._stop_camera_locked(clear_error=True)

            # Join previous worker thread outside the lock to guarantee it has fully terminated
            # and finished releasing its VideoCapture resource before we open a new handle.
            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=5.0)
                if thread_to_join.is_alive():
                    with cls._lock:
                        cls._is_connected = False
                        cls._error_message = "Previous camera capture worker failed to release within timeout."
                        return cls._get_info_locked()

            with cls._lock:
                # Attempt to open hardware webcam
                cap = cv2.VideoCapture(target_index)
                if not cap.isOpened():
                    cap.release()
                    cls._is_connected = False
                    cls._error_message = (
                        f"Unable to open camera index {target_index}. "
                        "Camera may be in use by another application, disconnected, or permission denied."
                    )
                    return cls._get_info_locked()

                # Configure resolution and FPS if supported by driver
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(settings.CAMERA_WIDTH))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(settings.CAMERA_HEIGHT))
                cap.set(cv2.CAP_PROP_FPS, float(settings.CAMERA_TARGET_FPS))

                # Read initial verification frame from real hardware
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    cls._is_connected = False
                    cls._error_message = f"Camera opened at index {target_index} but failed to capture initial frame."
                    return cls._get_info_locked()

                # Capture actual hardware capabilities reported by OpenCV
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (frame.shape[1] if frame is not None else settings.CAMERA_WIDTH)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (frame.shape[0] if frame is not None else settings.CAMERA_HEIGHT)
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0.0 or fps > 120.0:
                    fps = float(settings.CAMERA_TARGET_FPS)

                cls._generation += 1
                current_generation = cls._generation
                worker_stop_event = threading.Event()
                cls._worker_stop_event = worker_stop_event

                cls._capture = cap
                cls._camera_index = target_index
                cls._actual_width = width
                cls._actual_height = height
                cls._actual_fps = fps
                cls._latest_frame = frame
                cls._latest_frame_time = time.time()
                cls._consecutive_failures = 0
                cls._is_connected = True
                cls._error_message = None

                # Start background capture thread with explicit ownership of cap and worker_stop_event
                cls._thread = threading.Thread(
                    target=cls._capture_worker,
                    args=(cap, worker_stop_event, current_generation),
                    daemon=True,
                    name=f"CameraCaptureWorker-gen{current_generation}-idx{target_index}"
                )
                cls._thread.start()
                info = cls._get_info_locked()

        # Phase 3 hook: notify person detection service of active camera
        try:
            from app.services.person_detection_service import PersonDetectionService
            PersonDetectionService.on_camera_started()
        except Exception:
            pass

        # Phase 4 hook: notify fire detection service of active camera
        try:
            from app.services.fire_detection_service import FireDetectionService
            FireDetectionService.on_camera_started()
        except Exception:
            pass


        return info

    @classmethod
    def _capture_worker(cls, cap: cv2.VideoCapture, stop_event: threading.Event, generation: int):
        """
        Background worker thread continuously capturing frames from its assigned VideoCapture instance.
        Owns the specific 'cap' instance passed to it; does not rely on mutable class-level references.
        When permanent failure occurs (>30 consecutive failed reads), it releases its owned cap,
        cleans up class state if it is the current generation, and exits cleanly.
        """
        consecutive_failures = 0
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    with cls._lock:
                        if cls._generation == generation and not stop_event.is_set():
                            cls._latest_frame = frame
                            cls._latest_frame_time = time.time()
                            cls._consecutive_failures = 0
                            cls._frame_condition.notify_all()
                else:
                    consecutive_failures += 1
                    # If hardware read fails continuously for > 30 cycles, mark permanent failure
                    if consecutive_failures > 30:
                        with cls._lock:
                            if cls._generation == generation:
                                cls._is_connected = False
                                cls._error_message = "Camera disconnected or frame capture failed irrecoverably."
                                cls._latest_frame = None
                                cls._frame_condition.notify_all()
                        stop_event.set()
                        try:
                            from app.services.person_detection_service import PersonDetectionService
                            PersonDetectionService.on_camera_stopped()
                        except Exception:
                            pass
                        try:
                            from app.services.fire_detection_service import FireDetectionService
                            FireDetectionService.on_camera_stopped()
                        except Exception:
                            pass

                        break

                time.sleep(0.01)
        finally:
            # Guarantee hardware release for this worker's owned capture instance
            try:
                cap.release()
            except Exception:
                pass

            # Acquire lock only AFTER release has completed to guarantee that CameraService
            # never advertises the worker as gone or restartable before release is 100% finished.
            with cls._lock:
                if cls._generation == generation:
                    if cls._capture is cap:
                        cls._capture = None
                    if cls._thread is threading.current_thread():
                        cls._thread = None
                    if cls._worker_stop_event is stop_event:
                        cls._worker_stop_event = None
                    cls._frame_condition.notify_all()

    @classmethod
    def stop_camera(cls) -> Dict[str, Any]:
        """
        Safely stop video capture and release VideoCapture hardware resource.
        Multiple successive calls are idempotent and will not crash.
        Clears previous error state, returning clean 'Disconnected' status.
        """
        with cls._lifecycle_lock:
            thread_to_join = None
            with cls._lock:
                thread_to_join = cls._stop_camera_locked(clear_error=True)

            # Join capture thread outside lock to prevent deadlocks
            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=5.0)

            with cls._lock:
                info = cls._get_info_locked()

        # Phase 3 hook: notify person detection service of camera stop
        try:
            from app.services.person_detection_service import PersonDetectionService
            PersonDetectionService.on_camera_stopped()
        except Exception:
            pass
        # Phase 4 hook: notify fire detection service of camera stop
        try:
            from app.services.fire_detection_service import FireDetectionService
            FireDetectionService.on_camera_stopped()
        except Exception:
            pass

        return info

    @classmethod
    def _stop_camera_locked(cls, clear_error: bool = False) -> Optional[threading.Thread]:
        """Internal helper to shut down capture resources while holding lock."""
        if cls._worker_stop_event is not None:
            cls._worker_stop_event.set()

        cls._is_connected = False
        if clear_error:
            cls._error_message = None  # Explicit stop clears error state -> "Disconnected"
        cls._latest_frame = None
        cls._frame_condition.notify_all()

        old_thread = cls._thread
        # If no active worker thread is running, release capture immediately.
        # Otherwise, the worker thread will release its owned capture in its finally block.
        if old_thread is None or not old_thread.is_alive():
            cls._thread = None
            cls._worker_stop_event = None
            cap = cls._capture
            cls._capture = None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        return old_thread

    @classmethod
    def is_connected(cls) -> bool:
        """Return True if real webcam stream is active and connected."""
        with cls._lock:
            return cls._is_connected

    @classmethod
    def get_status(cls) -> str:
        """
        Return standard diagnostic status string:
        - 'Connected'
        - 'Disconnected'
        - 'Error: <details>'
        """
        with cls._lock:
            if cls._is_connected:
                return "Connected"
            if cls._error_message:
                return f"Error: {cls._error_message}"
            return "Disconnected"

    @classmethod
    def get_camera_info(cls) -> Dict[str, Any]:
        """Return complete status dictionary."""
        with cls._lock:
            return cls._get_info_locked()

    @classmethod
    def _get_info_locked(cls) -> Dict[str, Any]:
        return {
            "connected": cls._is_connected,
            "status": "Connected" if cls._is_connected else ("Error: " + cls._error_message if cls._error_message else "Disconnected"),
            "camera_index": cls._camera_index,
            "width": cls._actual_width,
            "height": cls._actual_height,
            "fps": cls._actual_fps,
            "error": cls._error_message
        }

    @classmethod
    def get_frame(cls) -> Optional[np.ndarray]:
        """
        Retrieve a copy of the latest real captured frame.
        Used by Phase 3 (Person Detector) and Phase 4 (Fire Detector) to consume
        the same frame without opening multiple VideoCapture instances.
        """
        with cls._lock:
            if not cls._is_connected or cls._latest_frame is None:
                return None
            return cls._latest_frame.copy()

    @classmethod
    def generate_frames(cls) -> Generator[bytes, None, None]:
        """
        Yield multipart MJPEG stream frames directly from OpenCV.
        Does not generate fake frames. Terminates cleanly if camera stops.
        """
        frame_interval = 1.0 / max(1.0, float(settings.CAMERA_TARGET_FPS))
        last_yield_time = 0.0

        while cls.is_connected():
            frame_to_encode = None
            with cls._lock:
                if not cls._is_connected:
                    break
                if cls._latest_frame is not None and cls._latest_frame_time >= last_yield_time:
                    frame_to_encode = cls._latest_frame.copy()
                    last_yield_time = time.time()

            if frame_to_encode is not None:
                # Phase 3 Presentation Layer: Overlay person bounding boxes onto stream frame copy
                try:
                    from app.services.person_detection_service import PersonDetectionService
                    frame_to_encode = PersonDetectionService.annotate_frame(frame_to_encode)
                except Exception:
                    pass

                # Phase 4 Presentation Layer: Overlay fire & smoke bounding boxes onto stream frame copy
                try:
                    from app.services.fire_detection_service import FireDetectionService
                    frame_to_encode = FireDetectionService.annotate_frame(frame_to_encode)
                except Exception:
                    pass

                # Encode actual real frame to JPEG
                success, buffer = cv2.imencode(
                    ".jpg",
                    frame_to_encode,
                    [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
                )
                if success:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )

            time.sleep(frame_interval)

    @classmethod
    def set_connected(cls, connected: bool, error: Optional[str] = None):
        """Compatibility helper for unit testing and manual status overrides."""
        with cls._lifecycle_lock:
            thread_to_join = None
            with cls._lock:
                cls._is_connected = connected
                cls._error_message = error
                if not connected:
                    thread_to_join = cls._stop_camera_locked(clear_error=False)

            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=5.0)

        if not connected:
            try:
                from app.services.person_detection_service import PersonDetectionService
                PersonDetectionService.on_camera_stopped()
            except Exception:
                pass
            try:
                from app.services.fire_detection_service import FireDetectionService
                FireDetectionService.on_camera_stopped()
            except Exception:
                pass

    @classmethod
    def get_generation(cls) -> int:
        """Return the current camera session generation ID."""
        with cls._lock:
            return cls._generation

    @classmethod
    def get_camera_index(cls) -> int:
        with cls._lock:
            return cls._camera_index

    @classmethod
    def set_camera_index(cls, index: int):
        with cls._lock:
            cls._camera_index = index
