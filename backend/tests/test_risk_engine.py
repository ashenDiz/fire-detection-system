"""
Comprehensive Unit Tests for the Multi-Source Risk Assessment Engine.
Tests all standard and edge scenarios mandated in university research specification.
"""

import pytest
from app.models.sensor import SensorReading
from app.services.risk_engine import RiskAssessmentService
from app.core.config import settings


def create_mock_reading(
    temperature: float = 24.0,
    humidity: float = 50.0,
    flame_level: float = 0.0,
    smoke_level: float = 2.0,
    gas_level: float = 5.0
) -> SensorReading:
    return SensorReading(
        id=1,
        temperature=temperature,
        humidity=humidity,
        flame_level=flame_level,
        smoke_level=smoke_level,
        gas_level=gas_level,
        input_source="manual_simulation"
    )


def test_scenario_normal_condition():
    """Scenario 1: Normal condition yields SAFE level and LOW emergency priority."""
    reading = create_mock_reading(temperature=24.0, humidity=55.0, smoke_level=3.0, gas_level=4.0, flame_level=0.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["risk_level"] == "SAFE"
    assert result["overall_risk_score"] < 30.0
    assert result["emergency_priority"] == "LOW"
    assert any("normal" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_high_temperature_only():
    """High temperature alone raises risk score and includes temperature explanation."""
    reading = create_mock_reading(temperature=65.0, humidity=40.0, smoke_level=5.0, gas_level=5.0, flame_level=0.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["temp_risk"] > 60.0
    assert result["overall_risk_score"] > 15.0
    assert any("temperature" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_high_gas_leak_only():
    """Scenario 2: High gas alone (e.g., 90%) results in WARNING / Gas Leak factor."""
    reading = create_mock_reading(temperature=26.0, humidity=50.0, smoke_level=5.0, gas_level=90.0, flame_level=0.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["gas_risk"] >= 85.0
    assert any("gas leak" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_high_smoke_only():
    """High smoke concentration alone escalates smoke sub-score."""
    reading = create_mock_reading(temperature=25.0, humidity=50.0, smoke_level=85.0, gas_level=10.0, flame_level=0.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["smoke_risk"] >= 85.0
    assert any("smoke" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_flame_indication_only():
    """Optical flame detection alone activates flame causal factor."""
    reading = create_mock_reading(temperature=28.0, humidity=50.0, smoke_level=10.0, gas_level=10.0, flame_level=85.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["flame_risk"] >= 80.0
    assert any("flame" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_multiple_abnormal_sensors():
    """Scenario 3: Multiple abnormal sensors (high temp, high smoke, high flame) -> HIGH RISK."""
    reading = create_mock_reading(temperature=60.0, humidity=35.0, smoke_level=70.0, gas_level=45.0, flame_level=70.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=None, people_count=0)

    assert result["risk_level"] in ["HIGH RISK", "CRITICAL"]
    assert result["overall_risk_score"] >= 65.0


def test_scenario_camera_fire_detection():
    """Scenario 4: High sensors + camera fire detection yields CRITICAL fire condition."""
    reading = create_mock_reading(temperature=75.0, humidity=25.0, smoke_level=90.0, gas_level=60.0, flame_level=90.0)
    result = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.92, people_count=0)

    assert result["risk_level"] == "CRITICAL"
    assert result["camera_fire_risk"] == 92.0
    assert any("fire detector confirms" in factor.lower() for factor in result["contributing_factors"])


def test_scenario_human_presence_escalation():
    """
    Scenario 5: Critical condition with people present escalates Emergency Priority to CRITICAL
    and includes specific human presence notice.
    """
    reading = create_mock_reading(temperature=80.0, humidity=20.0, smoke_level=95.0, gas_level=75.0, flame_level=95.0)
    result_with_people = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.95, people_count=4)
    result_zero_people = RiskAssessmentService.assess_risk(reading, camera_fire_conf=0.95, people_count=0)

    assert result_with_people["risk_level"] == "CRITICAL"
    assert result_with_people["emergency_priority"] == "CRITICAL"
    assert "4 people currently detected by camera" in result_with_people["alert_message"]

    assert result_zero_people["risk_level"] == "CRITICAL"
    assert result_zero_people["emergency_priority"] == "HIGH"
    assert "0 people currently detected by camera" in result_zero_people["alert_message"]


def test_boundary_values():
    """Test boundary inputs: absolute minimums (0.0) and maximums (100-120)."""
    # Minimum possible
    min_reading = create_mock_reading(temperature=0.0, humidity=0.0, smoke_level=0.0, gas_level=0.0, flame_level=0.0)
    min_res = RiskAssessmentService.assess_risk(min_reading, camera_fire_conf=0.0, people_count=0)
    assert min_res["overall_risk_score"] == 0.0
    assert min_res["risk_level"] == "SAFE"

    # Maximum possible
    max_reading = create_mock_reading(temperature=120.0, humidity=100.0, smoke_level=100.0, gas_level=100.0, flame_level=100.0)
    max_res = RiskAssessmentService.assess_risk(max_reading, camera_fire_conf=1.0, people_count=10)
    assert max_res["overall_risk_score"] == 100.0
    assert max_res["risk_level"] == "CRITICAL"
    assert max_res["emergency_priority"] == "CRITICAL"


def test_predefined_research_scenarios():
    """Ensure all predefined research scenarios calculate within expected risk brackets."""
    for sc_key, sc in settings.RESEARCH_SCENARIOS.items():
        reading = create_mock_reading(
            temperature=sc["temperature"],
            humidity=sc["humidity"],
            smoke_level=sc["smoke_level"],
            gas_level=sc["gas_level"],
            flame_level=sc["flame_level"]
        )
        res = RiskAssessmentService.assess_risk(reading)

        if sc_key == "NORMAL":
            assert res["risk_level"] == "SAFE"
        elif sc_key == "GAS_LEAK":
            assert res["risk_level"] in ["WARNING", "CAUTION"]
            assert res["gas_risk"] > 80.0
        elif sc_key == "POSSIBLE_FIRE":
            assert res["risk_level"] in ["WARNING", "HIGH RISK"]
        elif sc_key == "CRITICAL_FIRE":
            assert res["risk_level"] in ["HIGH RISK", "CRITICAL"]
            assert res["overall_risk_score"] >= 80.0
