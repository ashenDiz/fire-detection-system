from fastapi import APIRouter
from app.api.v1.system import router as system_router
from app.api.v1.sensors import router as sensors_router
from app.api.v1.risk import router as risk_router
from app.api.v1.scenarios import router as scenarios_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.events import router as events_router
from app.api.v1.camera import router as camera_router
from app.api.v1.detection import router as detection_router
from app.api.v1.emergency import router as emergency_router

api_v1_router = APIRouter()
api_v1_router.include_router(system_router, prefix="/system", tags=["System"])
api_v1_router.include_router(sensors_router, prefix="/sensors", tags=["Sensors"])
api_v1_router.include_router(risk_router, prefix="/risk", tags=["Risk Assessment"])
api_v1_router.include_router(scenarios_router, prefix="/scenarios", tags=["Research Scenarios"])
api_v1_router.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])
api_v1_router.include_router(events_router, prefix="/events", tags=["Detection Events"])
api_v1_router.include_router(camera_router, prefix="/camera", tags=["Camera"])
api_v1_router.include_router(detection_router, prefix="/detection", tags=["Detection"])
api_v1_router.include_router(emergency_router, prefix="/emergency", tags=["Emergency Response"])

