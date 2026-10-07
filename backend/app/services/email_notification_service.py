"""
Email Notification Service (Phase 5B).
Provides real SMTP email notifications on confirmed emergency state transitions.
Never blocks camera acquisition or YOLO inference.
Never crashes FastAPI on SMTP failure.
Never exposes credentials in logs or API responses.
"""

import logging
import smtplib
import threading
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Dict, Any, Optional, List

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """
    Thread-safe SMTP email notification manager for Phase 5B emergency alerts.
    """
    _lock = threading.RLock()
    _last_attempt_at: Optional[str] = None
    _last_success_at: Optional[str] = None
    _last_error: Optional[str] = None

    @classmethod
    def is_configured(cls) -> bool:
        """
        Check if required SMTP parameters are configured.
        Requires username and password when SMTP_REQUIRE_AUTH is True.
        """
        base_configured = bool(
            settings.SMTP_HOST
            and settings.SMTP_PORT
            and settings.EMERGENCY_EMAIL_FROM
            and settings.EMERGENCY_EMAIL_TO
        )
        if not base_configured:
            return False
        if getattr(settings, "SMTP_REQUIRE_AUTH", True):
            if not (settings.SMTP_USERNAME and settings.SMTP_PASSWORD):
                return False
        return True

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if email notifications are enabled in settings."""
        return bool(settings.EMAIL_NOTIFICATIONS_ENABLED)

    @classmethod
    def get_recipients(cls) -> List[str]:
        """Parse comma-separated recipients from EMERGENCY_EMAIL_TO."""
        raw = settings.EMERGENCY_EMAIL_TO or ""
        return [addr.strip() for addr in raw.split(",") if addr.strip()]

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """
        Return safe status dictionary.
        NEVER includes SMTP_PASSWORD or sensitive credentials.
        """
        with cls._lock:
            return {
                "enabled": cls.is_enabled(),
                "configured": cls.is_configured(),
                "last_attempt_at": cls._last_attempt_at,
                "last_success_at": cls._last_success_at,
                "last_error": cls._last_error,
            }

    @classmethod
    def build_email_content(cls, emergency_data: Dict[str, Any]) -> str:
        """
        Construct a truthful, comprehensive emergency report body.
        Clearly notes that environmental telemetry is manual_simulation in this prototype.
        """
        now_iso = emergency_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        state = emergency_data.get("state", "EMERGENCY_CONFIRMED")
        reason = emergency_data.get("reason", "Fire Emergency Condition Confirmed")
        fire_conf = emergency_data.get("fire_confidence")
        smoke_detected = emergency_data.get("visual_smoke_detected")
        smoke_conf = emergency_data.get("visual_smoke_confidence")
        people_count = emergency_data.get("visible_people_count")
        people_avail = emergency_data.get("people_detection_available", True)
        risk_level = emergency_data.get("risk_level")
        risk_score = emergency_data.get("risk_score")
        sensor_data = emergency_data.get("sensor_data")

        # Format visual fire confidence
        if fire_conf is not None:
            fire_line = f"Visual Fire Confidence: {fire_conf * 100.0:.1f}%"
        else:
            fire_line = "Visual Fire Confidence: Not Available"

        # Format visual smoke
        if smoke_detected is not None and smoke_conf is not None:
            smoke_line = f"Visual Smoke: {'Detected' if smoke_detected else 'None'} ({smoke_conf * 100.0:.1f}%)"
        else:
            smoke_line = "Visual Smoke: Not Available"

        # Format people count with strict truthfulness
        if not people_avail or people_count is None:
            people_line = "Visible Occupants: Person-detection status is currently unavailable; occupancy could not be verified."
        elif people_count == 0:
            people_line = "Visible Occupants: 0 people currently detected by camera. Actual occupancy is not confirmed."
        else:
            people_line = f"Visible Occupants: {people_count} person(s) currently detected in camera field of view."

        lines = [
            "============================================================",
            "🔥 CRITICAL FIRE EMERGENCY ALERT",
            "============================================================",
            f"Timestamp:       {now_iso}",
            f"Emergency State: {state}",
            f"Trigger Reason:  {reason}",
            "",
            "--- COMPUTER VISION TELEMETRY ---",
            fire_line,
            smoke_line,
            people_line,
        ]

        if risk_level is not None or risk_score is not None:
            lines.extend([
                "",
                "--- MULTI-MODAL RISK ASSESSMENT ---",
                f"Risk Level: {risk_level or 'N/A'}",
                f"Risk Score: {risk_score if risk_score is not None else 'N/A'}/100",
            ])

        if sensor_data:
            lines.extend([
                "",
                "--- SIMULATED ENVIRONMENTAL TELEMETRY ---",
                f"Temperature: {sensor_data.get('temperature', 'N/A')} °C",
                f"Humidity:    {sensor_data.get('humidity', 'N/A')} %",
                f"Smoke Level: {sensor_data.get('smoke_level', 'N/A')} %",
                f"Gas Level:   {sensor_data.get('gas_level', 'N/A')} %",
                f"Flame Level: {sensor_data.get('flame_level', 'N/A')} %",
                "Sensor Source: manual_simulation",
                "",
                "NOTE: Environmental telemetry in this research prototype is currently",
                "manual simulation and should not be interpreted as physical sensor",
                "hardware telemetry.",
            ])

        lines.extend([
            "",
            "============================================================",
            "System: Intelligent Multi-Modal Fire Detection & Emergency Alert",
            "============================================================",
        ])

        return "\n".join(lines)

    @classmethod
    def send_emergency_notification(cls, emergency_data: Dict[str, Any]) -> bool:
        """
        Send emergency notification email via SMTP.
        Returns True on success, False on skipped/error.
        Guarantees thread safety and failure isolation.
        """
        if not cls.is_enabled():
            logger.info("EmailNotificationService: Notifications disabled by configuration.")
            return False

        with cls._lock:
            cls._last_attempt_at = datetime.now(timezone.utc).isoformat()

            if not cls.is_configured():
                err = "SMTP configuration incomplete (check SMTP_HOST, PORT, FROM, TO)"
                cls._last_error = err
                logger.warning(f"EmailNotificationService: {err}")
                return False

            recipients = cls.get_recipients()
            if not recipients:
                err = "No valid recipients found in EMERGENCY_EMAIL_TO"
                cls._last_error = err
                logger.warning(f"EmailNotificationService: {err}")
                return False

        body_text = cls.build_email_content(emergency_data)

        msg = EmailMessage()
        msg["Subject"] = "[CRITICAL] Fire Emergency Detected"
        msg["From"] = settings.EMERGENCY_EMAIL_FROM
        msg["To"] = ", ".join(recipients)
        msg.set_content(body_text)

        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10.0) as server:
                if settings.SMTP_USE_TLS:
                    server.starttls()
                if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(msg)

            with cls._lock:
                cls._last_success_at = datetime.now(timezone.utc).isoformat()
                cls._last_error = None
            logger.info(f"EmailNotificationService: Emergency email sent successfully to {recipients}.")
            return True

        except Exception as exc:
            err_msg = f"SMTP transmission error: {str(exc)}"
            with cls._lock:
                cls._last_error = err_msg
            # Sanitize log output to avoid logging anything sensitive
            logger.error(f"EmailNotificationService failed to send email: {err_msg}")
            return False
