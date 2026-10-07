"""
Predefined Academic Research Test Scenarios.
Enables researchers to load repeatable, documented benchmark conditions into the simulator.
"""

from typing import List
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.sensor import ScenarioPreset

router = APIRouter()


@router.get("", response_model=List[ScenarioPreset])
def list_research_scenarios():
    """Retrieve predefined academic research benchmark scenarios."""
    scenarios: List[ScenarioPreset] = []
    for sc_id, sc_data in settings.RESEARCH_SCENARIOS.items():
        scenarios.append(
            ScenarioPreset(
                id=sc_id,
                name=sc_data["name"],
                description=sc_data["description"],
                temperature=sc_data["temperature"],
                humidity=sc_data["humidity"],
                smoke_level=sc_data["smoke_level"],
                gas_level=sc_data["gas_level"],
                flame_level=sc_data["flame_level"],
            )
        )
    return scenarios
