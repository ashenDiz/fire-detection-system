"""
Dedicated Risk Assessment and Multi-Source Sensor Fusion Service.
Implements normalization, weighted fusion, dynamic modality weighting,
transparent causal reasoning, single-hazard escalation (e.g. gas leak),
hazard signatures for alert deduplication, and emergency priority determination.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.sensor import SensorReading
from app.models.risk import RiskAssessment
from app.models.alert import Alert
from app.models.event import DetectionEvent
from app.services.camera_service import CameraService


class RiskAssessmentService:
    """
    Academic Multi-Modal Risk Fusion Engine.
    Combines environmental metrics (temperature, smoke, gas, flame) with visual AI metrics
    and human presence to compute an explainable, normalized 0-100 hazard risk score.
    """

    @staticmethod
    def normalize_temperature(temp: float) -> float:
        """
        Normalize ambient temperature (°C) to 0-100 risk sub-score.
        - <= 30°C: Safe baseline (0)
        - 30°C - 50°C: Elevated temperature (0 - 40)
        - 50°C - 70°C: Hazardous high heat (40 - 80)
        - > 70°C: Extreme combustion heat (80 - 100, capped at 100 for >= 100°C)
        """
        if temp <= settings.TEMP_SAFE_MAX:
            return 0.0
        elif temp <= settings.TEMP_ELEVATED_MAX:
            return ((temp - settings.TEMP_SAFE_MAX) / (settings.TEMP_ELEVATED_MAX - settings.TEMP_SAFE_MAX)) * 40.0
        elif temp <= settings.TEMP_HIGH_MAX:
            return 40.0 + ((temp - settings.TEMP_ELEVATED_MAX) / (settings.TEMP_HIGH_MAX - settings.TEMP_ELEVATED_MAX)) * 40.0
        else:
            score = 80.0 + ((temp - settings.TEMP_HIGH_MAX) / (settings.TEMP_CRITICAL - settings.TEMP_HIGH_MAX)) * 20.0
            return min(100.0, score)

    @staticmethod
    def normalize_smoke(smoke: float) -> float:
        """
        Normalize smoke obscuration level (0-100%) to 0-100 risk sub-score.
        - 0 - 15%: Normal atmospheric dust / ambient trace (0 - 15)
        - 15% - 40%: Developing haze / visible smoke (15 - 50)
        - 40% - 70%: Dense combustion smoke (50 - 85)
        - > 70%: Severe choking obscuration (85 - 100)
        """
        if smoke <= settings.SMOKE_BASELINE_MAX:
            return (smoke / settings.SMOKE_BASELINE_MAX) * 15.0
        elif smoke <= settings.SMOKE_ELEVATED_MAX:
            return 15.0 + ((smoke - settings.SMOKE_BASELINE_MAX) / (settings.SMOKE_ELEVATED_MAX - settings.SMOKE_BASELINE_MAX)) * 35.0
        elif smoke <= settings.SMOKE_HEAVY_MAX:
            return 50.0 + ((smoke - settings.SMOKE_ELEVATED_MAX) / (settings.SMOKE_HEAVY_MAX - settings.SMOKE_ELEVATED_MAX)) * 35.0
        else:
            return 85.0 + ((min(100.0, smoke) - settings.SMOKE_HEAVY_MAX) / (100.0 - settings.SMOKE_HEAVY_MAX)) * 15.0

    @staticmethod
    def normalize_gas(gas: float) -> float:
        """
        Normalize gas concentration (0-100%) to 0-100 risk sub-score.
        - 0 - 20%: Safe baseline (0 - 15)
        - 20% - 50%: Elevated volatile gas concentration (15 - 55)
        - 50% - 80%: Hazardous explosive/toxic build-up (55 - 85)
        - > 80%: Critical dangerous leak (85 - 100)
        """
        if gas <= settings.GAS_AMBIENT_MAX:
            return (gas / settings.GAS_AMBIENT_MAX) * 15.0
        elif gas <= settings.GAS_ELEVATED_MAX:
            return 15.0 + ((gas - settings.GAS_AMBIENT_MAX) / (settings.GAS_ELEVATED_MAX - settings.GAS_AMBIENT_MAX)) * 40.0
        elif gas <= settings.GAS_HAZARD_MAX:
            return 55.0 + ((gas - settings.GAS_ELEVATED_MAX) / (settings.GAS_HAZARD_MAX - settings.GAS_ELEVATED_MAX)) * 30.0
        else:
            return 85.0 + ((min(100.0, gas) - settings.GAS_HAZARD_MAX) / (100.0 - settings.GAS_HAZARD_MAX)) * 15.0

    @staticmethod
    def normalize_flame(flame: float) -> float:
        """
        Normalize optical flame sensor detection (0-100%) to 0-100 risk sub-score.
        - 0 - 20%: Ambient infrared fluctuations (0 - 15)
        - 20% - 60%: Detected intermittent radiant flame flicker (15 - 70)
        - > 60%: Direct continuous open flame exposure (70 - 100)
        """
        if flame <= settings.FLAME_NEGLIGIBLE_MAX:
            return (flame / settings.FLAME_NEGLIGIBLE_MAX) * 15.0
        elif flame <= settings.FLAME_MODERATE_MAX:
            return 15.0 + ((flame - settings.FLAME_NEGLIGIBLE_MAX) / (settings.FLAME_MODERATE_MAX - settings.FLAME_NEGLIGIBLE_MAX)) * 55.0
        else:
            return 70.0 + ((min(100.0, flame) - settings.FLAME_MODERATE_MAX) / (100.0 - settings.FLAME_MODERATE_MAX)) * 30.0

    @classmethod
    def determine_hazard_signature(
        cls,
        reading: SensorReading,
        camera_fire_conf: Optional[float] = None
    ) -> str:
        """
        Derive an explainable hazard/incident signature based on underlying causal telemetry.
        Prevents unrelated hazards (e.g. gas leak vs smoke hazard) from being conflated even if
        they share the same operational severity.
        """
        if camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIRMATION_THRESHOLD:
            return "CONFIRMED_FIRE_VISUAL"
        elif reading.flame_level >= 60.0 and (reading.temperature >= 50.0 or reading.smoke_level >= 40.0):
            return "MULTI_SENSOR_FIRE_RISK"
        elif reading.gas_level >= settings.GAS_ELEVATED_MAX:  # >= 50%
            return "GAS_LEAK"
        elif reading.flame_level > settings.FLAME_NEGLIGIBLE_MAX:  # > 20%
            return "OPTICAL_FLAME"
        elif reading.smoke_level >= settings.SMOKE_ELEVATED_MAX:  # >= 40%
            return "SMOKE_HAZARD"
        elif reading.temperature >= settings.TEMP_ELEVATED_MAX:  # >= 50°C
            return "HIGH_HEAT"
        else:
            return "GENERAL_ELEVATED_RISK"

    @classmethod
    def determine_risk_level(cls, overall_score: float) -> str:
        """Categorize risk score into discrete standard operational levels."""
        if overall_score <= settings.RISK_LEVEL_SAFE_MAX:
            return "SAFE"
        elif overall_score <= settings.RISK_LEVEL_CAUTION_MAX:
            return "CAUTION"
        elif overall_score <= settings.RISK_LEVEL_WARNING_MAX:
            return "WARNING"
        elif overall_score <= settings.RISK_LEVEL_HIGH_MAX:
            return "HIGH RISK"
        else:
            return "CRITICAL"

    @classmethod
    def evaluate_emergency_priority(
        cls,
        risk_level: str,
        people_count: Optional[int] = None,
        is_gas_leak: bool = False,
        camera_fire_confirmed: bool = False
    ) -> Tuple[str, str]:
        """
        Determine emergency priority and alert description based on risk level, occupants,
        and confirmed detection modality.
        
        CRITICAL RULE: During Phase 1 (no camera fire model), do NOT say 'Fire condition detected'
        for environmental sensor readings alone. Use 'Critical fire-risk condition indicated by environmental sensor fusion.'
        Only use wording equivalent to 'Fire detected' when an actual fire-detection modality confirms it.
        """
        if risk_level == "CRITICAL":
            if camera_fire_confirmed:
                if people_count is not None and people_count > 0:
                    priority = "CRITICAL"
                    message = f"CRITICAL EMERGENCY — Visual fire detection confirmed with {people_count} people currently detected by camera."
                elif people_count is not None and people_count == 0:
                    priority = "HIGH"
                    message = "HIGH PRIORITY — Visual fire detection confirmed; 0 people currently detected by camera."
                else:
                    # Unavailable person detection: do NOT downgrade to HIGH or claim area clear
                    priority = "CRITICAL"
                    message = "CRITICAL EMERGENCY — Visual fire detection confirmed. Person-detection status is currently unavailable; occupancy could not be determined."
            else:
                # Environmental sensor fusion only
                if people_count is not None and people_count > 0:
                    priority = "CRITICAL"
                    message = f"CRITICAL EMERGENCY — Critical fire-risk condition indicated by environmental sensor fusion with {people_count} people currently detected by camera."
                elif people_count is not None and people_count == 0:
                    priority = "HIGH"
                    message = "HIGH PRIORITY — Critical fire-risk condition indicated by environmental sensor fusion; 0 people currently detected by camera."
                else:
                    # Unavailable person detection: do NOT downgrade to HIGH or claim area clear
                    priority = "CRITICAL"
                    message = "CRITICAL EMERGENCY — Critical fire-risk condition indicated by environmental sensor fusion. Person-detection status is currently unavailable; occupancy could not be determined."
        elif risk_level == "HIGH RISK":
            if camera_fire_confirmed:
                if people_count is not None and people_count > 0:
                    priority = "HIGH"
                    message = f"HIGH RISK HAZARD — Visual fire detected with {people_count} people currently detected by camera."
                elif people_count is not None and people_count == 0:
                    priority = "MEDIUM"
                    message = "HIGH RISK HAZARD — Visual fire detected; 0 people currently detected by camera. Actual occupancy is not confirmed."
                else:
                    # Unavailable person detection: do NOT downgrade to MEDIUM or claim area clear
                    priority = "HIGH"
                    message = "HIGH RISK HAZARD — Visual fire detected. Person-detection status is currently unavailable; occupancy could not be determined."
            else:
                if people_count is not None and people_count > 0:
                    priority = "HIGH"
                    message = f"HIGH RISK HAZARD — High fire-risk condition indicated by environmental sensor fusion with {people_count} people currently detected by camera."
                elif people_count is not None and people_count == 0:
                    priority = "MEDIUM"
                    message = "HIGH RISK HAZARD — High fire-risk condition indicated by environmental sensor fusion; 0 people currently detected by camera. Actual occupancy is not confirmed."
                else:
                    # Unavailable person detection: do NOT downgrade to MEDIUM or claim area clear
                    priority = "HIGH"
                    message = "HIGH RISK HAZARD — High fire-risk condition indicated by environmental sensor fusion. Person-detection status is currently unavailable; occupancy could not be determined."
        elif risk_level == "WARNING":
            priority = "MEDIUM"
            if is_gas_leak:
                message = "WARNING — Hazardous gas concentration detected. Possible gas leak. Evacuate and ventilate area."
            else:
                message = "WARNING — Hazardous environmental readings detected. Immediate inspection advised."
        elif risk_level == "CAUTION":
            priority = "LOW"
            message = "CAUTION — Elevated environmental parameters observed. Continue active monitoring."
        else:
            priority = "LOW"
            message = "SAFE — Environmental parameters within standard baseline operating limits."

        return priority, message

    @classmethod
    def identify_contributing_factors(
        cls,
        temp: float,
        smoke: float,
        gas: float,
        flame: float,
        camera_fire_conf: Optional[float] = None,
        people_count: Optional[int] = None
    ) -> List[str]:
        """
        Generate transparent, evidence-based causal factors explaining the risk classification.
        """
        factors: List[str] = []

        # Temperature factors
        if temp >= settings.TEMP_HIGH_MAX:
            factors.append(f"Temperature severely elevated ({temp:.1f}°C) exceeding high risk threshold ({settings.TEMP_HIGH_MAX}°C)")
        elif temp >= settings.TEMP_ELEVATED_MAX:
            factors.append(f"Temperature elevated ({temp:.1f}°C) above normal operating baseline ({settings.TEMP_SAFE_MAX}°C)")

        # Smoke factors
        if smoke >= settings.SMOKE_HEAVY_MAX:
            factors.append(f"Dense smoke concentration ({smoke:.1f}%) indicates severe active combustion")
        elif smoke >= settings.SMOKE_ELEVATED_MAX:
            factors.append(f"Elevated smoke obscuration ({smoke:.1f}%) detected")

        # Gas factors
        if gas >= settings.GAS_HAZARD_MAX:
            factors.append(f"Dangerous gas concentration ({gas:.1f}%) exceeds explosive hazard threshold - Potential gas leak")
        elif gas >= settings.GAS_ELEVATED_MAX:
            factors.append(f"Volatile gas levels elevated ({gas:.1f}%)")

        # Flame factors
        if flame >= settings.FLAME_MODERATE_MAX:
            factors.append(f"Strong optical flame sensor detection ({flame:.1f}%) indicates direct open flame")
        elif flame > settings.FLAME_NEGLIGIBLE_MAX:
            factors.append(f"Intermittent flame sensor response detected ({flame:.1f}%)")

        # Camera fire factors (if present)
        if camera_fire_conf is not None and camera_fire_conf > 0.0:
            conf_pct = camera_fire_conf * 100.0
            if camera_fire_conf >= settings.FIRE_CONFIRMATION_THRESHOLD:
                factors.append(f"Computer vision fire detector confirms flame presence (Confidence: {conf_pct:.1f}%)")
            else:
                factors.append(f"Fire model produced a low-confidence indication ({conf_pct:.1f}%); below confirmation threshold.")

        # Human presence factor
        if people_count is not None and people_count > 0:
            factors.append(f"Occupancy detected: {people_count} people currently detected by camera in danger zone")
        elif people_count is None:
            factors.append("Person-detection status is currently unavailable; occupancy could not be verified.")

        if not factors:
            factors.append("All environmental metrics within normal baseline ranges.")

        return factors

    @classmethod
    def assess_risk(
        cls,
        reading: SensorReading,
        camera_fire_conf: Optional[float] = None,
        people_count: Optional[int] = None,
        visual_smoke_detected: Optional[bool] = None,
        visual_smoke_conf: Optional[float] = None
    ) -> Dict:
        """
        Execute complete multi-modal risk fusion calculation.
        Dynamically adjusts weights if camera AI is inactive (e.g., during Phase 1).
        Applies domain-specific fusion rules (e.g. acute gas leak elevation).
        """
        temp_score = cls.normalize_temperature(reading.temperature)
        smoke_score = cls.normalize_smoke(reading.smoke_level)
        gas_score = cls.normalize_gas(reading.gas_level)
        flame_score = cls.normalize_flame(reading.flame_level)

        # Modality weight configuration
        active_weights = {
            "temp": settings.WEIGHT_TEMPERATURE,
            "smoke": settings.WEIGHT_SMOKE,
            "gas": settings.WEIGHT_GAS,
            "flame": settings.WEIGHT_FLAME,
        }

        camera_score: Optional[float] = None
        if camera_fire_conf is not None:
            camera_score = min(100.0, max(0.0, camera_fire_conf * 100.0))
            active_weights["camera"] = settings.WEIGHT_CAMERA_FIRE

        # Re-normalize weights over available modalities so their sum is exactly 1.0
        weight_sum = sum(active_weights.values())
        norm_weights = {k: v / weight_sum for k, v in active_weights.items()}

        overall_score = (
            temp_score * norm_weights["temp"]
            + smoke_score * norm_weights["smoke"]
            + gas_score * norm_weights["gas"]
            + flame_score * norm_weights["flame"]
        )

        if camera_score is not None:
            overall_score += camera_score * norm_weights["camera"]

        # Domain Sensor Fusion Logic:
        # Scenario 2: Gas very high (>= 80% or hazard score >= 85), other values normal
        # Result must be at least WARNING (score >= 55) due to acute gas leak hazard.
        is_gas_leak = False
        if reading.gas_level >= settings.GAS_HAZARD_MAX:
            is_gas_leak = True
            # Escalate overall risk to at least WARNING range
            overall_score = max(overall_score, 55.0)

        # Dense smoke hazard (>= 70%) indicates active smoldering combustion
        # Escalate overall risk to at least WARNING range (score >= 50.0)
        if reading.smoke_level >= settings.SMOKE_HEAVY_MAX:
            overall_score = max(overall_score, 50.0)

        # Severe optical flame or critical fire indicators escalation
        if reading.flame_level >= 85.0 and reading.smoke_level >= 70.0 and reading.temperature >= 60.0:
            overall_score = max(overall_score, 85.0)  # CRITICAL

        # Clamp between 0.0 and 100.0 and round
        overall_score = round(max(0.0, min(100.0, overall_score)), 2)
        risk_level = cls.determine_risk_level(overall_score)
        
        camera_fire_confirmed = (
            camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIRMATION_THRESHOLD
        )
        priority, alert_message = cls.evaluate_emergency_priority(
            risk_level,
            people_count,
            is_gas_leak=is_gas_leak,
            camera_fire_confirmed=camera_fire_confirmed
        )
        factors = cls.identify_contributing_factors(
            temp=reading.temperature,
            smoke=reading.smoke_level,
            gas=reading.gas_level,
            flame=reading.flame_level,
            camera_fire_conf=camera_fire_conf,
            people_count=people_count
        )

        return {
            "overall_risk_score": overall_score,
            "risk_level": risk_level,
            "temp_risk": round(temp_score, 2),
            "smoke_risk": round(smoke_score, 2),
            "gas_risk": round(gas_score, 2),
            "flame_risk": round(flame_score, 2),
            "camera_fire_risk": round(camera_score, 2) if camera_score is not None else None,
            "fire_detection_available": (camera_fire_conf is not None),
            "fire_confidence": camera_fire_conf,
            "camera_fire_confidence": camera_fire_conf,
            "visual_smoke_detected": visual_smoke_detected,
            "visual_smoke_confidence": visual_smoke_conf,
            "people_detected": people_count,
            "people_detection_available": (people_count is not None),
            "emergency_priority": priority,
            "alert_message": alert_message,
            "contributing_factors": factors,
        }

    @classmethod
    def evaluate_and_record(
        cls,
        reading: SensorReading,
        db: Session,
        camera_fire_conf: Optional[float] = None,
        people_count: Optional[int] = None,
        visual_smoke_detected: Optional[bool] = None,
        visual_smoke_conf: Optional[float] = None
    ) -> Tuple[RiskAssessment, Optional[Alert]]:
        """
        Evaluate risk, persist assessment to database, manage state-aware alerts with hazard signatures
        (avoiding alert spamming and preserving causality), and log detection state changes to DetectionEvent.
        """
        # Determine previous risk level to detect state transitions
        prev_assessment = db.query(RiskAssessment).order_by(RiskAssessment.created_at.desc()).first()
        prev_risk_level = prev_assessment.risk_level if prev_assessment else None

        assessment_data = cls.assess_risk(
            reading,
            camera_fire_conf=camera_fire_conf,
            people_count=people_count,
            visual_smoke_detected=visual_smoke_detected,
            visual_smoke_conf=visual_smoke_conf
        )
        is_avail = assessment_data["people_detection_available"]
        stored_people_count = people_count if is_avail and people_count is not None else 0
        is_fire_avail = assessment_data["fire_detection_available"]

        db_assessment = RiskAssessment(
            sensor_reading_id=reading.id,
            overall_risk_score=assessment_data["overall_risk_score"],
            risk_level=assessment_data["risk_level"],
            temp_risk=assessment_data["temp_risk"],
            smoke_risk=assessment_data["smoke_risk"],
            gas_risk=assessment_data["gas_risk"],
            flame_risk=assessment_data["flame_risk"],
            camera_fire_risk=assessment_data["camera_fire_risk"],
            fire_detection_available=is_fire_avail,
            fire_confidence=camera_fire_conf,
            visual_smoke_detected=visual_smoke_detected,
            visual_smoke_confidence=visual_smoke_conf,
            people_detected=stored_people_count,
            people_detection_available=is_avail,
            emergency_priority=assessment_data["emergency_priority"],
            contributing_factors=json.dumps(assessment_data["contributing_factors"]),
            created_at=datetime.now(timezone.utc)
        )
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)

        # Derive explainable hazard signature
        hazard_sig = cls.determine_hazard_signature(reading, camera_fire_conf)

        # Alert Generation with Hazard Signature & Anti-Spam Traceability
        db_alert: Optional[Alert] = None
        new_alert_created = False
        if assessment_data["risk_level"] != "SAFE":
            alert_type = assessment_data["risk_level"]
            if alert_type == "HIGH RISK":
                alert_type = "HIGH"

            SEVERITY_ORDER = {"CAUTION": 1, "WARNING": 2, "HIGH": 3, "CRITICAL": 4}
            current_severity = SEVERITY_ORDER.get(alert_type, 1)

            # Query for an active unacknowledged alert matching this specific hazard signature
            existing_unacked = db.query(Alert).filter(
                Alert.acknowledged == False,
                Alert.hazard_signature == hazard_sig
            ).order_by(Alert.created_at.desc()).first()

            if existing_unacked:
                prev_severity = SEVERITY_ORDER.get(existing_unacked.alert_type, 1)
                if current_severity > prev_severity:
                    # Condition meaningfully escalated -> create new escalated alert record
                    db_alert = Alert(
                        hazard_signature=hazard_sig,
                        sensor_reading_id=reading.id,
                        risk_assessment_id=db_assessment.id,
                        risk_score=assessment_data["overall_risk_score"],
                        risk_level=assessment_data["risk_level"],
                        alert_type=alert_type,
                        message=assessment_data["alert_message"],
                        people_detected=stored_people_count,
                        people_detection_available=is_avail,
                        fire_detection_available=is_fire_avail,
                        fire_detected=(
                            camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD
                        ),
                        fire_confidence=camera_fire_conf,
                        acknowledged=False,
                        created_at=datetime.now(timezone.utc)
                    )
                    db.add(db_alert)
                    db.commit()
                    db.refresh(db_alert)
                    new_alert_created = True
                else:
                    # Same continuing hazard with equal or decreased severity ->
                    # Policy: Re-use the existing active alert ID to prevent notification spam,
                    # but synchronize its CURRENT state fields consistently with latest assessment.
                    existing_unacked.sensor_reading_id = reading.id
                    existing_unacked.risk_assessment_id = db_assessment.id
                    existing_unacked.risk_score = assessment_data["overall_risk_score"]
                    existing_unacked.risk_level = assessment_data["risk_level"]
                    existing_unacked.alert_type = alert_type
                    existing_unacked.message = assessment_data["alert_message"]
                    existing_unacked.people_detected = stored_people_count
                    existing_unacked.people_detection_available = is_avail
                    existing_unacked.fire_detection_available = is_fire_avail
                    existing_unacked.fire_detected = (
                        camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD
                    )
                    existing_unacked.fire_confidence = camera_fire_conf
                    db.commit()
                    db.refresh(existing_unacked)
                    db_alert = existing_unacked
            else:
                # No active unacknowledged alert for this hazard_sig (or previous acknowledged) -> create new alert
                db_alert = Alert(
                    hazard_signature=hazard_sig,
                    sensor_reading_id=reading.id,
                    risk_assessment_id=db_assessment.id,
                    risk_score=assessment_data["overall_risk_score"],
                    risk_level=assessment_data["risk_level"],
                    alert_type=alert_type,
                    message=assessment_data["alert_message"],
                    people_detected=stored_people_count,
                    people_detection_available=is_avail,
                    fire_detection_available=is_fire_avail,
                    fire_detected=(
                        camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD
                    ),
                    fire_confidence=camera_fire_conf,
                    acknowledged=False,
                    created_at=datetime.now(timezone.utc)
                )
                db.add(db_alert)
                db.commit()
                db.refresh(db_alert)
                new_alert_created = True

        # State-Change Detection Event Logging (Section 21)
        # Log when risk level changes or a new alert occurs
        risk_level_changed = (prev_risk_level != assessment_data["risk_level"])
        if risk_level_changed or new_alert_created:
            camera_status = CameraService.get_status()
            detection_event = DetectionEvent(
                fire_detection_available=is_fire_avail,
                fire_detected=(
                    camera_fire_conf is not None and camera_fire_conf >= settings.FIRE_CONFIDENCE_THRESHOLD
                ),
                fire_confidence=camera_fire_conf,
                people_count=stored_people_count,
                people_detection_available=is_avail,
                risk_score=assessment_data["overall_risk_score"],
                risk_level=assessment_data["risk_level"],
                camera_status=camera_status,
                created_at=datetime.now(timezone.utc)
            )
            db.add(detection_event)
            db.commit()

        return db_assessment, db_alert
