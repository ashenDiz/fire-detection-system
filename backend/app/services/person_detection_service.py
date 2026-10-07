"""
Person Detection and Visible Person Counting Service (Phase 3).
Consumes raw OpenCV frames from CameraService.get_frame().
Never opens a secondary cv2.VideoCapture instance (source=0 is forbidden).
Runs official Ultralytics COCO-pretrained object detection model (yolo26n.pt)
in a dedicated, thread-safe background inference worker.
Maintains presentation-layer overlay bounding boxes without modifying the raw frame.
"""

import os
import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from app.core.config import settings
from app.services.camera_service import CameraService


class PersonDetectionService:
    """
    Dedicated, thread-safe service managing the YOLO person-detection lifecycle.
    Keeps person AI completely decoupled from camera hardware acquisition.
    """
    _lock: threading.RLock = threading.RLock()
    _lifecycle_lock: threading.RLock = threading.RLock()
    _model: Any = None
    _model_status: str = "Not Loaded"  # "Not Loaded", "Loaded", "Error: <reason>"
    _model_name: str = settings.PERSON_MODEL_PATH
    _person_class_id: Optional[int] = None

    _worker_thread: Optional[threading.Thread] = None
    _stop_event: Optional[threading.Event] = None
    _worker_generation: int = 0
    _last_inference_error: Optional[str] = None

    _latest_detections: List[Dict[str, Any]] = []
    _latest_person_count: int = 0
    _last_inference_time: float = 0.0
    _last_inference_latency_ms: float = 0.0
    _inference_fps: float = 0.0
    _camera_session_id: int = 0

    @classmethod
    def resolve_model_path(cls, model_path: Optional[str] = None) -> str:
        """Resolve model path relative to project structure or use configured path."""
        path = model_path or settings.PERSON_MODEL_PATH
        if os.path.isabs(path) and os.path.exists(path):
            return path

        # Check repository root / backend root
        candidate_paths = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), path),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return path

    @classmethod
    def initialize_model(cls, model_path: Optional[str] = None) -> bool:
        """
        Load YOLO detection model once and dynamically identify the 'person' class ID.
        Does not reload model per frame.
        Guarantees that model failure does not affect camera capture.
        """
        with cls._lock:
            target_path = cls.resolve_model_path(model_path)
            try:
                from ultralytics import YOLO

                model = YOLO(target_path)

                # Dynamically locate the class whose name equals 'person'
                person_id = None
                if hasattr(model, "names") and isinstance(model.names, dict):
                    for cid, name in model.names.items():
                        if str(name).strip().lower() == "person":
                            person_id = int(cid)
                            break

                if person_id is None:
                    cls._model = None
                    cls._person_class_id = None
                    cls._model_status = "Error: 'person' class not found in model class mapping"
                    cls._latest_detections = []
                    cls._latest_person_count = 0
                    cls._last_inference_time = 0.0
                    return False

                cls._model = model
                cls._person_class_id = person_id
                cls._model_name = os.path.basename(target_path)
                cls._model_status = "Loaded"
                return True
            except Exception as e:
                cls._model = None
                cls._person_class_id = None
                cls._model_status = f"Error: {str(e)}"
                cls._latest_detections = []
                cls._latest_person_count = 0
                cls._last_inference_time = 0.0
                return False

    @classmethod
    def set_mock_model(cls, mock_model: Any, person_class_id: int = 0):
        """Helper for isolated unit testing without downloading weights or requiring GPU."""
        with cls._lock:
            cls._model = mock_model
            cls._person_class_id = person_class_id
            cls._model_status = "Loaded"

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Return True only if model is verified loaded and ready for inference."""
        with cls._lock:
            return cls._model is not None and cls._model_status == "Loaded"

    @classmethod
    def get_model_status(cls) -> str:
        """
        Return exact model operational status:
        - 'Loaded'
        - 'Not Loaded'
        - 'Error: <reason>'
        """
        with cls._lock:
            return cls._model_status

    @classmethod
    def start_worker(cls, join_timeout: float = 2.0) -> bool:
        """
        Start dedicated background inference worker thread if model is loaded.
        Thread-safe: guarantees no duplicate workers can be created.
        Never starts a replacement worker while a previous worker is still alive or cleaning up.
        """
        with cls._lifecycle_lock:
            thread_to_join = None
            with cls._lock:
                # If an active, non-stopping worker is already running, return True
                if cls._worker_thread is not None and cls._worker_thread.is_alive():
                    if cls._stop_event is not None and not cls._stop_event.is_set():
                        return True
                    # Existing worker was requested to stop; capture reference to join
                    thread_to_join = cls._worker_thread

            # Join outside service state lock
            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=join_timeout)

            with cls._lock:
                # If previous worker is STILL alive (e.g. join timeout occurred), do NOT advertise
                # the worker as gone and do NOT start a replacement worker!
                if cls._worker_thread is not None and cls._worker_thread.is_alive():
                    return False

                if not cls.is_model_loaded():
                    return False

                cls._worker_generation += 1
                gen = cls._worker_generation
                stop_event = threading.Event()
                cls._stop_event = stop_event
                cls._worker_thread = threading.Thread(
                    target=cls._inference_worker,
                    args=(stop_event, gen),
                    daemon=True,
                    name=f"PersonInferenceWorker-gen{gen}"
                )
                cls._worker_thread.start()
                return True

    @classmethod
    def stop_worker(cls, timeout: float = 2.0):
        """
        Safely stop background inference worker and wait for thread exit.
        Retains the worker reference while it is still alive so replacement workers cannot race.
        Only clears worker/event references after termination.
        """
        thread_to_join = None
        with cls._lifecycle_lock:
            with cls._lock:
                if cls._stop_event is not None:
                    cls._stop_event.set()
                thread_to_join = cls._worker_thread
                cls.clear_detections()
                # DO NOT clear cls._worker_thread here; retain it while alive!

            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=timeout)

            with cls._lock:
                # Only clear worker/event references after termination, and only if they still belong to that worker
                if thread_to_join is not None and not thread_to_join.is_alive():
                    if cls._worker_thread is thread_to_join:
                        cls._worker_thread = None
                        cls._stop_event = None

    @classmethod
    def _inference_worker(cls, stop_event: threading.Event, generation: int = 1):
        """
        Controlled background worker pulling newest frames from CameraService.get_frame().
        Executes YOLO inference at configured FPS (5-10 FPS) without blocking video stream.
        """
        target_fps = max(1.0, float(settings.PERSON_INFERENCE_FPS))
        frame_interval = 1.0 / target_fps

        try:
            while not stop_event.is_set():
                loop_start = time.perf_counter()

                # Verify camera connection status
                if not CameraService.is_connected():
                    with cls._lock:
                        cls._latest_detections = []
                        cls._latest_person_count = 0
                        cls._last_inference_time = 0.0
                    time.sleep(0.1)
                    continue

                current_session = CameraService.get_generation()
                raw_frame = CameraService.get_frame()

                if raw_frame is None:
                    time.sleep(0.05)
                    continue

                # Run YOLO person inference on real NumPy frame (NOT source=0)
                try:
                    model = cls._model
                    person_cid = cls._person_class_id
                    if model is None or person_cid is None:
                        time.sleep(0.2)
                        continue

                    inf_start = time.perf_counter()
                    results = model.predict(
                        source=raw_frame,
                        classes=[person_cid],
                        conf=settings.PERSON_CONFIDENCE_THRESHOLD,
                        imgsz=settings.PERSON_IMAGE_SIZE,
                        device=settings.PERSON_DEVICE,
                        verbose=False
                    )
                    inf_end = time.perf_counter()
                    latency_ms = (inf_end - inf_start) * 1000.0

                    parsed_detections: List[Dict[str, Any]] = []
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None:
                            for box in boxes:
                                conf = float(box.conf[0])
                                cid = int(box.cls[0])
                                if cid == person_cid and conf >= settings.PERSON_CONFIDENCE_THRESHOLD:
                                    raw_coords = box.xyxy[0]
                                    if hasattr(raw_coords, "tolist"):
                                        raw_coords = raw_coords.tolist()
                                    xyxy = [round(float(c), 1) for c in raw_coords]
                                    parsed_detections.append({
                                        "bbox": xyxy,
                                        "confidence": round(conf, 4),
                                        "class_name": "person"
                                    })

                    with cls._lock:
                        # Stale session check: verify camera is still connected and generation hasn't changed
                        if not CameraService.is_connected() or CameraService.get_generation() != current_session:
                            cls._latest_detections = []
                            cls._latest_person_count = 0
                            cls._last_inference_time = 0.0
                        else:
                            cls._latest_detections = parsed_detections
                            cls._latest_person_count = len(parsed_detections)
                            cls._last_inference_time = time.time()
                            cls._last_inference_latency_ms = round(latency_ms, 2)
                            cls._camera_session_id = current_session
                            cls._inference_fps = round(1000.0 / max(1.0, latency_ms), 1)
                            cls._last_inference_error = None

                except Exception as e:
                    # Inference failure must NOT crash FastAPI or capture worker
                    # CRITICAL: Invalidate freshness immediately!
                    with cls._lock:
                        cls._latest_detections = []
                        cls._latest_person_count = 0
                        cls._last_inference_time = 0.0
                        cls._last_inference_error = str(e)
                    time.sleep(0.2)

                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.01, frame_interval - elapsed)
                time.sleep(sleep_time)
        finally:
            with cls._lock:
                if cls._worker_generation == generation:
                    if cls._worker_thread is threading.current_thread():
                        cls._worker_thread = None
                    if cls._stop_event is stop_event:
                        cls._stop_event = None

    @classmethod
    def clear_detections(cls):
        """Reset detection state immediately."""
        with cls._lock:
            cls._latest_detections = []
            cls._latest_person_count = 0
            cls._last_inference_time = 0.0
            cls._last_inference_error = None

    @classmethod
    def on_camera_started(cls):
        """Lifecycle hook: triggered when CameraService successfully connects."""
        cls.clear_detections()
        if cls.is_model_loaded():
            cls.start_worker()

    @classmethod
    def on_camera_stopped(cls):
        """Lifecycle hook: triggered when CameraService disconnects or errors."""
        cls.clear_detections()

    @classmethod
    def is_detection_fresh(cls) -> bool:
        """Check if latest detection is from the active camera session, model is loaded, and within stale timeout."""
        with cls._lock:
            if not cls.is_model_loaded():
                return False
            if not CameraService.is_connected():
                return False
            if cls._last_inference_time <= 0:
                return False
            if CameraService.get_generation() != cls._camera_session_id:
                return False
            if time.time() - cls._last_inference_time > settings.PERSON_STALE_TIMEOUT:
                return False
            return True

    @classmethod
    def get_fresh_person_count(cls) -> Optional[int]:
        """
        Return the visible person count if detector is loaded, camera is active, and data is fresh.
        Returns None if person detection is unavailable (camera off, model error, or stale data).
        """
        with cls._lock:
            if not cls.is_model_loaded():
                return None
            if not cls.is_detection_fresh():
                return None
            return cls._latest_person_count

    @classmethod
    def get_fresh_detections(cls) -> List[Dict[str, Any]]:
        """Return fresh detections list, or empty list if stale or camera inactive."""
        with cls._lock:
            if not cls.is_detection_fresh():
                return []
            return list(cls._latest_detections)

    @classmethod
    def annotate_frame(cls, frame: np.ndarray) -> np.ndarray:
        """
        Presentation layer overlay: draws bounding boxes and confidence labels on a copy of the frame.
        Does not mutate the original raw frame stored by CameraService.
        """
        detections = cls.get_fresh_detections()
        if not detections or frame is None:
            return frame

        h_img, w_img = frame.shape[:2]
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1 = max(0, min(w_img - 1, int(bbox[0])))
            y1 = max(0, min(h_img - 1, int(bbox[1])))
            x2 = max(0, min(w_img - 1, int(bbox[2])))
            y2 = max(0, min(h_img - 1, int(bbox[3])))
            conf_pct = int(det.get("confidence", 0.0) * 100)

            # Draw bounding box (vibrant emerald green BGR: (46, 204, 113))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (113, 204, 46), 2)

            # Draw label box
            label = f"Person {conf_pct}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_box_top = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, y_box_top), (x1 + tw + 8, y_box_top + th + 8), (113, 204, 46), -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y_box_top + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return frame

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Diagnostic status for /api/v1/detection/person/status."""
        with cls._lock:
            last_dt = (
                datetime.fromtimestamp(cls._last_inference_time, tz=timezone.utc)
                if cls._last_inference_time > 0
                else None
            )
            is_fresh = cls.is_detection_fresh()
            return {
                "model_loaded": cls.is_model_loaded(),
                "status": cls._model_status,
                "model_name": cls._model_name,
                "enabled": cls._worker_thread is not None and cls._worker_thread.is_alive(),
                "confidence_threshold": settings.PERSON_CONFIDENCE_THRESHOLD,
                "people_count": cls._latest_person_count if is_fresh else None,
                "detection_available": is_fresh,
                "last_inference_at": last_dt,
                "inference_fps": cls._inference_fps,
                "inference_latency_ms": cls._last_inference_latency_ms,
                "device": settings.PERSON_DEVICE,
            }

    @classmethod
    def get_latest(cls) -> Dict[str, Any]:
        """Latest detection data for /api/v1/detection/person/latest."""
        with cls._lock:
            last_dt = (
                datetime.fromtimestamp(cls._last_inference_time, tz=timezone.utc)
                if cls._last_inference_time > 0
                else None
            )
            is_fresh = cls.is_detection_fresh()
            return {
                "model_status": cls._model_status,
                "model_loaded": cls.is_model_loaded(),
                "camera_connected": CameraService.is_connected(),
                "people_count": cls._latest_person_count if is_fresh else None,
                "detection_available": is_fresh,
                "detections": list(cls._latest_detections) if is_fresh else [],
                "last_inference_at": last_dt,
                "inference_latency_ms": cls._last_inference_latency_ms,
                "stale": not is_fresh,
            }
