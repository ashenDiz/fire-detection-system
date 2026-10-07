"""
Emergency Response and Confirmation Service (Phase 5B).
Consumes semantic outputs from FireDetectionService, PersonDetectionService,
and recent simulated environmental telemetry.
Never directly accesses camera hardware devices or YOLO inference.
Maintains state machine: NORMAL -> MONITORING -> EMERGENCY_CONFIRMED.
Handles temporal confirmation, multi-modal corroboration, anti-flicker clearing,
and triggers real email notification on emergency episode state transitions.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.services.fire_detection_service import FireDetectionService
from app.services.person_detection_service import PersonDetectionService
from app.services.email_notification_service import EmailNotificationService

logger = logging.getLogger(__name__)


class EmergencyResponseService:
    """
    Thread-safe Phase 5B Emergency Response and Confirmation Service.
    """
    _lifecycle_lock: threading.RLock = threading.RLock()
    _lock: threading.RLock = threading.RLock()

    # State machine variables
    _state: str = "NORMAL"  # "NORMAL", "MONITORING", "EMERGENCY_CONFIRMED"
    _priority: str = "LOW"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    _reason: str = "Normal operating conditions. No fire candidate detected."
    _confirmation_source: Optional[str] = None  # "VISUAL_TEMPORAL", "MULTIMODAL", "SENSOR_CRITICAL", or None
    _last_updated_at: str = datetime.now(timezone.utc).isoformat()

    # Temporal confirmation tracker (stores timestamps of unique observations >= FIRE_CONFIRMATION_THRESHOLD)
    _temporal_hits: List[float] = []

    # Anti-flicker clearing counter (counts consecutive distinct fresh negative 0.0 samples)
    _consecutive_negative_samples: int = 0

    # Stable inference identity tracking to reject duplicate monitor polls
    _last_processed_fire_observation_id: Optional[Any] = None

    # Timestamp tracking for sensor-critical assessments to enforce newer non-critical updates
    _critical_assessment_time: Optional[float] = None

    # Anti-spam email episode tracker
    _episode_email_sent: bool = False

    # Cached last observation context
    _last_fire_avail: bool = False
    _last_raw_fire_detected: Optional[bool] = None
    _last_fire_conf: Optional[float] = None
    _last_visual_smoke_detected: Optional[bool] = None
    _last_visual_smoke_conf: Optional[float] = None
    _last_people_avail: bool = False
    _last_visible_people: Optional[int] = None
    _last_sensor_context_available: bool = False
    _last_sensor_input_source: Optional[str] = None
    _last_observation_id: Optional[int] = None
    _last_observation_timestamp: Optional[float] = None

    # Background monitor worker lifecycle
    _monitor_thread: Optional[threading.Thread] = None
    _stop_event: Optional[threading.Event] = None
    _monitor_running: bool = False

    @classmethod
    def reset_state(cls):
        """Reset state machine completely for deterministic testing."""
        with cls._lock:
            cls._state = "NORMAL"
            cls._priority = "LOW"
            cls._reason = "Normal operating conditions. No fire candidate detected."
            cls._confirmation_source = None
            cls._last_updated_at = datetime.now(timezone.utc).isoformat()
            cls._temporal_hits = []
            cls._consecutive_negative_samples = 0
            cls._last_processed_fire_observation_id = None
            cls._critical_assessment_time = None
            cls._episode_email_sent = False
            cls._last_fire_avail = False
            cls._last_raw_fire_detected = None
            cls._last_fire_conf = None
            cls._last_visual_smoke_detected = None
            cls._last_visual_smoke_conf = None
            cls._last_people_avail = False
            cls._last_visible_people = None
            cls._last_sensor_context_available = False
            cls._last_sensor_input_source = None
            cls._last_observation_id = None
            cls._last_observation_timestamp = None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """
        Return transparent snapshot of emergency response status.
        Never exposes fake values.
        """
        with cls._lock:
            return {
                "state": cls._state,
                "priority": cls._priority,
                "reason": cls._reason,
                "confirmation_source": cls._confirmation_source,
                "fire_detection_available": cls._last_fire_avail,
                "raw_fire_detected": cls._last_raw_fire_detected,
                "fire_confidence": cls._last_fire_conf,
                "confirmation_threshold": settings.FIRE_CONFIRMATION_THRESHOLD,
                "temporal_hits": len(cls._temporal_hits),
                "people_detection_available": cls._last_people_avail,
                "visible_people_count": cls._last_visible_people,
                "visual_smoke_detected": cls._last_visual_smoke_detected,
                "visual_smoke_confidence": cls._last_visual_smoke_conf,
                "sensor_context_available": cls._last_sensor_context_available,
                "sensor_input_source": cls._last_sensor_input_source,
                "observation_id": cls._last_observation_id,
                "observation_timestamp": cls._last_observation_timestamp,
                "last_updated_at": cls._last_updated_at,
            }

    @classmethod
    def evaluate_step(
        cls,
        fire_summary: Dict[str, Any],
        person_count: Optional[int] = None,
        sensor_reading: Optional[Any] = None,
        risk_assessment: Optional[Any] = None,
        current_timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Deterministic state evaluation step.
        Can be invoked directly by unit tests or periodically by background monitor.
        Enforces unique inference identity, source-aware clearing, and strict timestamp validity.
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        with cls._lock:
            fire_avail = bool(fire_summary.get("detection_available", False))
            fire_conf = fire_summary.get("fire_confidence")
            raw_fire_detected = fire_summary.get("fire_detected")
            smoke_detected = fire_summary.get("smoke_detected")
            smoke_conf = fire_summary.get("smoke_confidence")

            # Separate stable inference identity and timestamp:
            # observation_id is used ONLY for duplicate-observation detection.
            # observation_timestamp is used ONLY for the emergency API observation_timestamp field.
            observation_id = fire_summary.get("inference_id")
            raw_obs_timestamp = fire_summary.get("inference_timestamp")
            observation_timestamp = float(raw_obs_timestamp) if raw_obs_timestamp is not None else None

            # Check if this is a genuinely NEW fire inference observation.
            # Fail closed when stable inference identity is unavailable:
            # fresh fire semantic result with NO inference_id identity
            # MUST NOT count as a new temporal confirmation hit or negative clearing sample.
            # Do NOT fabricate identity from monitor poll time / current_timestamp.
            is_new_visual_obs = False
            if fire_avail and observation_id is not None:
                if observation_id != cls._last_processed_fire_observation_id:
                    is_new_visual_obs = True
                    cls._last_processed_fire_observation_id = observation_id
            elif not fire_avail:
                # Detector unavailable: reset observation ID tracking so subsequent fresh inference is caught
                cls._last_processed_fire_observation_id = None

            people_avail = (person_count is not None) or PersonDetectionService.is_model_loaded()
            visible_people = person_count

            # Check environmental sensor reading (STRICT: missing created_at is NOT fresh)
            sensor_context_available = False
            sensor_input_source = None
            sensor_corroborated = False

            if sensor_reading is not None:
                reading_time = getattr(sensor_reading, "created_at", None)
                if reading_time is None:
                    # Missing created_at must NOT be considered fresh
                    sensor_context_available = False
                else:
                    if reading_time.tzinfo is None:
                        reading_dt = reading_time.replace(tzinfo=timezone.utc)
                    else:
                        reading_dt = reading_time
                    age = (datetime.now(timezone.utc) - reading_dt).total_seconds()
                    if 0.0 <= age <= settings.EMERGENCY_SENSOR_MAX_AGE_SECONDS:
                        sensor_context_available = True
                        sensor_input_source = getattr(sensor_reading, "input_source", "manual_simulation")
                        flame = float(getattr(sensor_reading, "flame_level", 0.0))
                        smoke = float(getattr(sensor_reading, "smoke_level", 0.0))
                        temp = float(getattr(sensor_reading, "temperature", 0.0))

                        # Multi-modal corroboration criteria:
                        # flame_level >= 60 OR (smoke_level >= 40 AND temperature >= 50)
                        if flame >= 60.0 or (smoke >= 40.0 and temp >= 50.0):
                            sensor_corroborated = True
                    else:
                        sensor_context_available = False

            # Check sensor-only critical assessment (STRICT: missing created_at is NOT valid evidence)
            sensor_critical = False
            has_valid_recent_assessment = False
            latest_assessment_epoch = None
            latest_critical_epoch = None

            if risk_assessment is not None:
                assessment_time = getattr(risk_assessment, "created_at", None)
                if assessment_time is not None:
                    if assessment_time.tzinfo is None:
                        ass_dt = assessment_time.replace(tzinfo=timezone.utc)
                    else:
                        ass_dt = assessment_time
                    age = (datetime.now(timezone.utc) - ass_dt).total_seconds()
                    if 0.0 <= age <= settings.EMERGENCY_SENSOR_MAX_AGE_SECONDS:
                        has_valid_recent_assessment = True
                        latest_assessment_epoch = ass_dt.timestamp()
                        if getattr(risk_assessment, "risk_level", "") == "CRITICAL":
                            sensor_critical = True
                            latest_critical_epoch = latest_assessment_epoch

            # Cache latest observation metrics
            cls._last_fire_avail = fire_avail
            cls._last_raw_fire_detected = raw_fire_detected
            cls._last_fire_conf = fire_conf
            cls._last_visual_smoke_detected = smoke_detected
            cls._last_visual_smoke_conf = smoke_conf
            cls._last_people_avail = (person_count is not None)
            cls._last_visible_people = visible_people
            cls._last_sensor_context_available = sensor_context_available
            cls._last_sensor_input_source = sensor_input_source
            cls._last_observation_id = int(observation_id) if (fire_avail and isinstance(observation_id, int)) else None
            cls._last_observation_timestamp = observation_timestamp if fire_avail else None
            cls._last_updated_at = now_iso

            # --- Temporal observation window update ---
            # Purge expired hits outside the rolling window
            cls._temporal_hits = [
                t for t in cls._temporal_hits
                if (now - t) <= settings.FIRE_CONFIRMATION_WINDOW_SECONDS
            ]

            # Register fresh observation ONLY IF this is a genuinely NEW inference identity
            if is_new_visual_obs:
                if fire_conf is not None and fire_conf >= settings.FIRE_CONFIRMATION_THRESHOLD:
                    cls._temporal_hits.append(now)

            # Determine temporal confirmation
            temporal_confirmed = len(cls._temporal_hits) >= settings.FIRE_CONFIRMATION_MIN_HITS

            # Determine multi-modal candidate corroboration
            has_raw_candidate = (fire_avail and fire_conf is not None and fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD)
            multimodal_confirmed = has_raw_candidate and sensor_corroborated

            # --- Anti-Flicker Clearing Logic for Confirmed Emergencies ---
            if cls._state == "EMERGENCY_CONFIRMED":
                # Increment negative counter ONLY on distinct new visual no-fire observations
                if is_new_visual_obs:
                    if fire_conf == 0.0:
                        cls._consecutive_negative_samples += 1
                    elif fire_conf is not None and fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD:
                        cls._consecutive_negative_samples = 0

                # Escalate confirmation source if visual confirmation arises during sensor emergency
                if cls._confirmation_source == "SENSOR_CRITICAL":
                    if temporal_confirmed:
                        cls._confirmation_source = "VISUAL_TEMPORAL"
                    elif multimodal_confirmed:
                        cls._confirmation_source = "MULTIMODAL"

                # Check source-aware clearing conditions
                cleared = False
                if cls._confirmation_source in ("VISUAL_TEMPORAL", "MULTIMODAL", None):
                    # Visual/Multimodal emergencies require 5 distinct fresh negative visual samples and no active sensor-critical
                    if cls._consecutive_negative_samples >= settings.FIRE_CLEAR_MIN_FRESH_NEGATIVE_SAMPLES and not sensor_critical:
                        cleared = True
                        cls._reason = "Normal operating conditions. Previous visual emergency cleared."

                elif cls._confirmation_source == "SENSOR_CRITICAL":
                    # Sensor-only critical emergency does NOT require camera observations.
                    # It clears ONLY when a newer, recent, valid non-critical risk assessment proves safety.
                    # Missing or stale telemetry never clears it.
                    if (
                        has_valid_recent_assessment
                        and not sensor_critical
                        and latest_assessment_epoch is not None
                        and (cls._critical_assessment_time is None or latest_assessment_epoch > cls._critical_assessment_time)
                    ):
                        cleared = True
                        cls._reason = "Normal operating conditions. Previous sensor-critical emergency cleared by newer non-critical telemetry."

                if cleared:
                    cls._state = "NORMAL"
                    cls._priority = "LOW"
                    cls._confirmation_source = None
                    cls._temporal_hits = []
                    cls._consecutive_negative_samples = 0
                    cls._critical_assessment_time = None
                    cls._episode_email_sent = False
                    return cls.get_status()

            # --- State Transitions from Non-Confirmed State ---
            new_state = cls._state
            new_priority = cls._priority
            new_reason = cls._reason
            triggered_new_emergency = False

            if temporal_confirmed:
                new_state = "EMERGENCY_CONFIRMED"
                cls._confirmation_source = "VISUAL_TEMPORAL"
                new_reason = (
                    f"Visual fire emergency confirmed ({len(cls._temporal_hits)} hits "
                    f">= {settings.FIRE_CONFIRMATION_THRESHOLD * 100:.0f}% in "
                    f"{settings.FIRE_CONFIRMATION_WINDOW_SECONDS:.1f}s window)."
                )
                if cls._state != "EMERGENCY_CONFIRMED":
                    triggered_new_emergency = True

            elif multimodal_confirmed:
                new_state = "EMERGENCY_CONFIRMED"
                cls._confirmation_source = "MULTIMODAL"
                new_reason = (
                    f"Visual fire candidate ({fire_conf * 100:.1f}%) corroborated by "
                    f"recent simulated environmental telemetry."
                )
                if cls._state != "EMERGENCY_CONFIRMED":
                    triggered_new_emergency = True

            elif sensor_critical:
                new_state = "EMERGENCY_CONFIRMED"
                cls._confirmation_source = "SENSOR_CRITICAL"
                if latest_critical_epoch is not None:
                    cls._critical_assessment_time = latest_critical_epoch
                new_reason = (
                    "Critical fire-risk condition indicated by manual simulated "
                    "environmental telemetry."
                )
                if cls._state != "EMERGENCY_CONFIRMED":
                    triggered_new_emergency = True

            elif cls._state != "EMERGENCY_CONFIRMED":
                # Candidate / Monitoring state
                if has_raw_candidate:
                    new_state = "MONITORING"
                    cls._confirmation_source = None
                    new_reason = (
                        f"Possible visual fire detected ({fire_conf * 100:.1f}%) — "
                        f"monitoring for confirmation."
                    )
                else:
                    new_state = "NORMAL"
                    cls._confirmation_source = None
                    new_reason = "Normal operating conditions. No fire candidate detected."

            # Calculate Priority
            if new_state == "EMERGENCY_CONFIRMED":
                if visible_people is not None and visible_people > 0:
                    new_priority = "CRITICAL"
                elif visible_people is not None and visible_people == 0:
                    new_priority = "HIGH"
                else:
                    new_priority = "CRITICAL"  # Unavailable person detection: do not downgrade
            elif new_state == "MONITORING":
                new_priority = "HIGH" if (visible_people is not None and visible_people > 0) else "MEDIUM"
            else:
                new_priority = "LOW"

            cls._state = new_state
            cls._priority = new_priority
            cls._reason = new_reason

            # --- Email Anti-Spam Notification Policy ---
            # Send email ONLY on transition into EMERGENCY_CONFIRMED
            if triggered_new_emergency and not cls._episode_email_sent:
                cls._episode_email_sent = True
                emergency_payload = {
                    "state": new_state,
                    "reason": new_reason,
                    "timestamp": now_iso,
                    "fire_confidence": fire_conf,
                    "visual_smoke_detected": smoke_detected,
                    "visual_smoke_confidence": smoke_conf,
                    "visible_people_count": visible_people,
                    "people_detection_available": (person_count is not None),
                    "risk_level": getattr(risk_assessment, "risk_level", None) if risk_assessment else None,
                    "risk_score": getattr(risk_assessment, "overall_risk_score", None) if risk_assessment else None,
                    "sensor_data": {
                        "temperature": getattr(sensor_reading, "temperature", None),
                        "humidity": getattr(sensor_reading, "humidity", None),
                        "smoke_level": getattr(sensor_reading, "smoke_level", None),
                        "gas_level": getattr(sensor_reading, "gas_level", None),
                        "flame_level": getattr(sensor_reading, "flame_level", None),
                    } if sensor_reading else None,
                }
                # Dispatch email in non-blocking thread
                threading.Thread(
                    target=EmailNotificationService.send_emergency_notification,
                    args=(emergency_payload,),
                    daemon=True,
                    name="EmergencyEmailDispatch"
                ).start()

            return cls.get_status()

    @classmethod
    def start_monitor(cls):
        """
        Start single background monitor worker safely at FastAPI startup.
        Guarantees strictly one worker, preventing duplicates or deadlocks.
        """
        with cls._lifecycle_lock:
            with cls._lock:
                if cls._monitor_running and cls._monitor_thread is not None and cls._monitor_thread.is_alive():
                    return
                # Clean up stale worker reference if dead
                if cls._monitor_thread is not None and not cls._monitor_thread.is_alive():
                    cls._monitor_thread = None
                    cls._stop_event = None

                cls._stop_event = threading.Event()
                cls._monitor_running = True
                cls._monitor_thread = threading.Thread(
                    target=cls._monitor_worker,
                    args=(cls._stop_event,),
                    daemon=True,
                    name="EmergencyMonitorWorker"
                )
                cls._monitor_thread.start()
                logger.info("EmergencyResponseService background monitor worker started.")

    @classmethod
    def stop_monitor(cls, timeout: float = 2.0):
        """
        Stop background monitor worker cleanly at FastAPI shutdown.
        Strict lock hierarchy: sets stop event, joins outside _lock.
        """
        thread_to_join = None
        with cls._lifecycle_lock:
            with cls._lock:
                if cls._stop_event is not None:
                    cls._stop_event.set()
                cls._monitor_running = False
                thread_to_join = cls._monitor_thread

            if thread_to_join is not None and thread_to_join.is_alive() and threading.current_thread() != thread_to_join:
                thread_to_join.join(timeout=timeout)

            with cls._lock:
                if thread_to_join is not None and not thread_to_join.is_alive():
                    if cls._monitor_thread is thread_to_join:
                        cls._monitor_thread = None
                        cls._stop_event = None
                logger.info("EmergencyResponseService background monitor worker stopped.")

    @classmethod
    def _monitor_worker(cls, stop_event: threading.Event):
        """
        Periodic background monitor loop.
        Consumes semantic outputs without performing YOLO or VideoCapture operations.
        """
        from app.database.session import SessionLocal
        from app.models.sensor import SensorReading
        from app.models.risk import RiskAssessment

        while not stop_event.is_set():
            try:
                # 1. Fetch fresh semantic summaries
                fire_summary = FireDetectionService.get_fresh_summary()
                person_count = PersonDetectionService.get_fresh_person_count()

                # 2. Query latest sensor reading and risk assessment from DB
                latest_reading = None
                latest_assessment = None
                db = None
                try:
                    db = SessionLocal()
                    latest_reading = db.query(SensorReading).order_by(SensorReading.created_at.desc()).first()
                    latest_assessment = db.query(RiskAssessment).order_by(RiskAssessment.created_at.desc()).first()
                except Exception as db_err:
                    logger.debug(f"Emergency monitor DB query skipped: {db_err}")
                finally:
                    if db is not None:
                        db.close()

                # 3. Evaluate step
                cls.evaluate_step(
                    fire_summary=fire_summary,
                    person_count=person_count,
                    sensor_reading=latest_reading,
                    risk_assessment=latest_assessment,
                    current_timestamp=time.time()
                )

            except Exception as loop_err:
                logger.error(f"Error in emergency monitor worker loop: {loop_err}", exc_info=True)

            # Sleep ~0.4s (2.5 checks per second)
            stop_event.wait(0.4)
