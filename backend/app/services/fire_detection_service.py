"""
Fire and Smoke Detection Service (Phase 4).
Consumes raw OpenCV frames from CameraService.get_frame().
Never opens a secondary VideoCapture hardware instance (hardware acquisition is strictly forbidden).
Runs custom-trained D-Fire YOLO model (backend/models/best.pt)
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


class FireDetectionService:
    """
    Dedicated, thread-safe service managing the custom D-Fire YOLO fire/smoke detection lifecycle.
    Keeps visual fire/smoke AI completely decoupled from camera hardware acquisition.
    """
    _lock: threading.RLock = threading.RLock()
    _lifecycle_lock: threading.RLock = threading.RLock()
    _model: Any = None
    _model_status: str = "Not Loaded"  # "Not Loaded", "Loaded", "Error: <reason>"
    _model_name: str = settings.FIRE_MODEL_PATH
    _fire_class_id: Optional[int] = None
    _smoke_class_id: Optional[int] = None

    _worker_thread: Optional[threading.Thread] = None
    _stop_event: Optional[threading.Event] = None
    _worker_generation: int = 0
    _last_inference_error: Optional[str] = None

    _latest_detections: List[Dict[str, Any]] = []
    _fire_detected: Optional[bool] = None
    _smoke_detected: Optional[bool] = None
    _fire_confidence: Optional[float] = None
    _smoke_confidence: Optional[float] = None
    _last_inference_time: float = 0.0
    _inference_sequence: int = 0
    _last_inference_latency_ms: float = 0.0
    _inference_fps: float = 0.0
    _camera_session_id: int = 0

    @classmethod
    def resolve_model_path(cls, model_path: Optional[str] = None) -> str:
        """Resolve model path relative to project structure or use configured path."""
        path = model_path or settings.FIRE_MODEL_PATH
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
        Load custom D-Fire YOLO model once and dynamically identify 'fire' and 'smoke' class IDs.
        Does not reload model per frame.
        Guarantees that model failure does not affect camera capture or crash FastAPI.
        """
        with cls._lock:
            target_path = cls.resolve_model_path(model_path)
            try:
                from ultralytics import YOLO

                model = YOLO(target_path)

                # Dynamically locate class IDs whose names match 'fire' and 'smoke' (case-insensitive)
                fire_id = None
                smoke_id = None
                if hasattr(model, "names") and isinstance(model.names, dict):
                    for cid, name in model.names.items():
                        norm = str(name).strip().lower()
                        if norm == "fire":
                            fire_id = int(cid)
                        elif norm == "smoke":
                            smoke_id = int(cid)

                if fire_id is None or smoke_id is None:
                    missing = []
                    if fire_id is None:
                        missing.append("'fire'")
                    if smoke_id is None:
                        missing.append("'smoke'")
                    cls._model = None
                    cls._fire_class_id = None
                    cls._smoke_class_id = None
                    cls._model_status = f"Error: Required class(es) {', '.join(missing)} not found in model class mapping"
                    cls.clear_detections()
                    return False

                cls._model = model
                cls._fire_class_id = fire_id
                cls._smoke_class_id = smoke_id
                cls._model_name = os.path.basename(target_path)
                cls._model_status = "Loaded"
                return True
            except Exception as e:
                cls._model = None
                cls._fire_class_id = None
                cls._smoke_class_id = None
                cls._model_status = f"Error: {str(e)}"
                cls.clear_detections()
                return False

    @classmethod
    def set_mock_model(cls, mock_model: Any, fire_class_id: int = 1, smoke_class_id: int = 0):
        """Helper for isolated unit testing without downloading weights or requiring GPU."""
        with cls._lock:
            cls._model = mock_model
            cls._fire_class_id = fire_class_id
            cls._smoke_class_id = smoke_class_id
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
        Strict lock hierarchy: _lifecycle_lock -> _lock.
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
                # If previous worker is STILL alive, do not create duplicate
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
                    name=f"FireInferenceWorker-gen{gen}"
                )
                cls._worker_thread.start()
                return True

    @classmethod
    def stop_worker(cls, timeout: float = 2.0):
        """
        Safely stop background inference worker and wait for thread exit.
        Retains the worker reference while it is still alive so replacement workers cannot race.
        Only clears worker/event references after termination.
        Strict lock hierarchy: _lifecycle_lock -> _lock.
        """
        thread_to_join = None
        with cls._lifecycle_lock:
            with cls._lock:
                if cls._stop_event is not None:
                    cls._stop_event.set()
                thread_to_join = cls._worker_thread
                cls.clear_detections()

            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=timeout)

            with cls._lock:
                if thread_to_join is not None and not thread_to_join.is_alive():
                    if cls._worker_thread is thread_to_join:
                        cls._worker_thread = None
                        cls._stop_event = None

    @classmethod
    def _inference_worker(cls, stop_event: threading.Event, generation: int = 1):
        """
        Controlled background worker pulling newest frames from CameraService.get_frame().
        Executes YOLO inference at configured FPS (3.0 FPS default) without blocking video stream.
        """
        target_fps = max(1.0, float(settings.FIRE_INFERENCE_FPS))
        frame_interval = 1.0 / target_fps

        try:
            while not stop_event.is_set():
                loop_start = time.perf_counter()

                # Verify camera connection status
                if not CameraService.is_connected():
                    with cls._lock:
                        cls.clear_detections()
                    time.sleep(0.1)
                    continue

                current_session = CameraService.get_generation()
                raw_frame = CameraService.get_frame()

                if raw_frame is None:
                    time.sleep(0.05)
                    continue

                with cls._lock:
                    model = cls._model
                    fire_cid = cls._fire_class_id
                    smoke_cid = cls._smoke_class_id

                if model is None or fire_cid is None or smoke_cid is None:
                    time.sleep(0.2)
                    continue

                try:
                    inf_start = time.perf_counter()
                    results = model.predict(
                        source=raw_frame,
                        classes=[smoke_cid, fire_cid],
                        conf=settings.FIRE_CONFIDENCE_THRESHOLD,
                        imgsz=settings.FIRE_IMAGE_SIZE,
                        device=settings.FIRE_DEVICE,
                        verbose=False
                    )
                    inf_end = time.perf_counter()
                    latency_ms = (inf_end - inf_start) * 1000.0

                    parsed_detections: List[Dict[str, Any]] = []
                    fire_confs: List[float] = []
                    smoke_confs: List[float] = []

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None:
                            for box in boxes:
                                conf = float(box.conf[0])
                                cid = int(box.cls[0])
                                if conf >= settings.FIRE_CONFIDENCE_THRESHOLD:
                                    if cid == fire_cid:
                                        raw_coords = box.xyxy[0]
                                        if hasattr(raw_coords, "tolist"):
                                            raw_coords = raw_coords.tolist()
                                        xyxy = [round(float(c), 1) for c in raw_coords]
                                        parsed_detections.append({
                                            "bbox": xyxy,
                                            "confidence": round(conf, 4),
                                            "class_name": "fire"
                                        })
                                        fire_confs.append(conf)
                                    elif cid == smoke_cid:
                                        raw_coords = box.xyxy[0]
                                        if hasattr(raw_coords, "tolist"):
                                            raw_coords = raw_coords.tolist()
                                        xyxy = [round(float(c), 1) for c in raw_coords]
                                        parsed_detections.append({
                                            "bbox": xyxy,
                                            "confidence": round(conf, 4),
                                            "class_name": "smoke"
                                        })
                                        smoke_confs.append(conf)

                    # Calculate summary values based on highest confidences
                    fire_detected = len(fire_confs) > 0
                    fire_conf = round(max(fire_confs), 4) if fire_detected else 0.0

                    smoke_detected = len(smoke_confs) > 0
                    smoke_conf = round(max(smoke_confs), 4) if smoke_detected else 0.0

                    with cls._lock:
                        invalid_result = (
                            stop_event.is_set()
                            or generation != cls._worker_generation
                            or cls._stop_event is not stop_event
                            or not CameraService.is_connected()
                            or CameraService.get_generation() != current_session
                            or not cls.is_model_loaded()
                            or cls._model is not model
                        )

                        if invalid_result:
                            cls.clear_detections()
                        else:
                            cls._latest_detections = parsed_detections
                            cls._fire_detected = fire_detected
                            cls._smoke_detected = smoke_detected
                            cls._fire_confidence = fire_conf
                            cls._smoke_confidence = smoke_conf
                            cls._last_inference_time = time.time()
                            cls._inference_sequence += 1
                            cls._last_inference_latency_ms = round(latency_ms, 2)
                            cls._camera_session_id = current_session
                            cls._inference_fps = round(1000.0 / max(1.0, latency_ms), 1)
                            cls._last_inference_error = None

                except Exception as e:
                    # Inference failure must NOT crash FastAPI or CameraService
                    # CRITICAL: Invalidate freshness immediately!
                    with cls._lock:
                        cls.clear_detections()
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
        """Reset detection state immediately to unavailable."""
        with cls._lock:
            cls._latest_detections = []
            cls._fire_detected = None
            cls._smoke_detected = None
            cls._fire_confidence = None
            cls._smoke_confidence = None
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
        """Check if latest detection is from active camera session, model is loaded, and within stale timeout."""
        with cls._lock:
            if not cls.is_model_loaded():
                return False
            if not CameraService.is_connected():
                return False
            if cls._last_inference_time <= 0:
                return False
            if CameraService.get_generation() != cls._camera_session_id:
                return False
            if time.time() - cls._last_inference_time > settings.FIRE_STALE_TIMEOUT:
                return False
            return True

    @classmethod
    def get_fresh_detections(cls) -> List[Dict[str, Any]]:
        """Return fresh detections list, or empty list if stale or camera inactive."""
        with cls._lock:
            if not cls.is_detection_fresh():
                return []
            return list(cls._latest_detections)

    @classmethod
    def get_fresh_summary(cls) -> Dict[str, Any]:
        """
        Return fire/smoke detection summary if fresh and available.
        Distinguishes valid zero result from unavailable detector.
        """
        with cls._lock:
            if not cls.is_detection_fresh():
                return {
                    "detection_available": False,
                    "fire_detected": None,
                    "smoke_detected": None,
                    "fire_confidence": None,
                    "smoke_confidence": None,
                    "inference_timestamp": None,
                    "inference_id": None,
                }
            return {
                "detection_available": True,
                "fire_detected": cls._fire_detected,
                "smoke_detected": cls._smoke_detected,
                "fire_confidence": cls._fire_confidence,
                "smoke_confidence": cls._smoke_confidence,
                "inference_timestamp": cls._last_inference_time,
                "inference_id": cls._inference_sequence,
            }

    @classmethod
    def annotate_frame(cls, frame: np.ndarray) -> np.ndarray:
        """
        Presentation layer overlay: draws fire and smoke bounding boxes on a copy of the frame.
        Does not mutate the original raw frame stored by CameraService.
        Uses visually distinct bounding boxes:
        - Fire: vibrant crimson / red (BGR: 20, 20, 230)
        - Smoke: amber / orange (BGR: 0, 140, 255)
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
            cname = det.get("class_name", "").lower()

            if cname == "fire":
                color = (20, 20, 230)  # Crimson Red
                label = f"Fire {conf_pct}%"
            elif cname == "smoke":
                color = (0, 140, 255)  # Amber / Dark Orange
                label = f"Smoke {conf_pct}%"
            else:
                color = (0, 200, 200)
                label = f"{cname.capitalize()} {conf_pct}%"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_box_top = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, y_box_top), (x1 + tw + 8, y_box_top + th + 8), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y_box_top + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Diagnostic status for /api/v1/detection/fire/status."""
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
                "confidence_threshold": settings.FIRE_CONFIDENCE_THRESHOLD,
                "detection_available": is_fresh,
                "fire_detected": cls._fire_detected if is_fresh else None,
                "smoke_detected": cls._smoke_detected if is_fresh else None,
                "fire_confidence": cls._fire_confidence if is_fresh else None,
                "smoke_confidence": cls._smoke_confidence if is_fresh else None,
                "last_inference_at": last_dt,
                "inference_fps": cls._inference_fps,
                "inference_latency_ms": cls._last_inference_latency_ms,
                "device": settings.FIRE_DEVICE,
            }

    @classmethod
    def get_latest(cls) -> Dict[str, Any]:
        """Latest detection data for /api/v1/detection/fire/latest."""
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
                "detection_available": is_fresh,
                "fire_detected": cls._fire_detected if is_fresh else None,
                "smoke_detected": cls._smoke_detected if is_fresh else None,
                "fire_confidence": cls._fire_confidence if is_fresh else None,
                "smoke_confidence": cls._smoke_confidence if is_fresh else None,
                "detections": list(cls._latest_detections) if is_fresh else [],
                "last_inference_at": last_dt,
                "inference_latency_ms": cls._last_inference_latency_ms,
                "stale": not is_fresh,
            }
