# Intelligent Multi-Modal Fire Detection and Emergency Alert System
### Using Sensor Fusion and Computer Vision

An academic and research prototype combining environmental sensor telemetry, multi-modal risk assessment, and computer vision for early industrial, warehouse, and laboratory fire hazard detection.

---

## 1. Project Overview

This prototype investigates multi-modal sensor fusion algorithms for fire safety. Rather than relying on a single detection modality (which is prone to false alarms from dust or steam), the system combines:
1. **Environmental Sensor Telemetry**: Ambient temperature (°C), relative humidity (%), smoke obscuration (%), combustible/toxic gas concentration (%), and optical flame presence (%).
2. **Multi-Source Risk Assessment Engine**: A centralized weighted normalization engine that calculates an explainable 0–100 risk score and operational risk classifications (`SAFE`, `CAUTION`, `WARNING`, `HIGH RISK`, `CRITICAL`).
3. **Computer Vision Inference**:
   - **Phase 2 (Completed)**: Real OpenCV webcam acquisition, thread-safe background frame distribution, and live MJPEG video streaming.
   - **Phase 3 (Completed)**: Real YOLO person detection (`yolo26n.pt`), visible person counting, presentation-layer bounding box rendering, and stale detection protection.
   - **Phase 4 (Upcoming)**: Dedicated custom YOLO fire detection model (`models/fire_detector.pt`).
4. **Emergency Priority Engine**: Contextual escalation of emergency priority based on detected occupants in the hazard zone.
5. **Incident Logging & Alert System**: Event persistence in SQLite with alert state tracking and deduplication.

---

## 2. Research Transparency & Integrity Rules

In accordance with academic standards:
- **No Synthetic Sensor Readings**: Sensor metrics are strictly provided via user input in the manual simulator or through predefined research test scenarios.
- **No Fake Webcam or AI Detections**: When hardware or model weights are not present, the system transparently reports `"Camera Disconnected"` and `"Fire AI Model Not Loaded"`. No simulated, prerecorded, or looped video is ever rendered.
- **Visible Occupancy Limitations (Crucial Academic Distinction)**:
  - The displayed metric is **"People Currently Detected by Camera"** — representing individuals currently visible in the active camera field of view.
  - It is **NOT** guaranteed to equal true room or building occupancy because individuals may be:
    - Outside the physical camera field of view (FOV),
    - Occluded by partitions, furniture, or structural pillars,
    - Too distant or too small to resolve at the input resolution,
    - In low-illumination or high-contrast shadow conditions,
    - Or missed by the object detector.
  - No facial recognition, biometric identity profiling, or persistent tracking across time is performed in Phase 3.
- **Modality State Separation**: Connecting a webcam does **not** imply that fire or people have been detected, nor that AI models are active. Camera status and AI model status remain strictly decoupled.

---

## 3. System Architecture & Component Design

```
fire-detection-system/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint & lifespan shutdown
│   │   ├── core/
│   │   │   └── config.py            # Centralized weights, boundaries, camera settings
│   │   ├── database/
│   │   │   └── session.py           # SQLite connection & PRAGMA foreign_keys hook
│   │   ├── models/                  # SQLAlchemy ORM entities
│   │   │   ├── sensor.py            # SensorReading table
│   │   │   ├── risk.py              # RiskAssessment table
│   │   │   ├── alert.py             # Alert table (with hazard_signature tracking)
│   │   │   └── event.py             # DetectionEvent table
│   │   ├── schemas/                 # Pydantic validation schemas
│   │   │   ├── sensor.py
│   │   │   ├── risk.py
│   │   │   ├── system.py
│   │   │   └── camera.py            # Phase 2 Camera schemas
│   │   ├── services/
│   │   │   ├── sensor_provider.py   # Abstract SensorDataProvider & ManualSensorProvider
│   │   │   ├── risk_engine.py       # Multi-source fusion & causal reasoning
│   │   │   ├── system_service.py    # System health & diagnostic checks
│   │   │   └── camera_service.py    # OpenCV VideoCapture & MJPEG generator
│   │   └── api/
│   │       └── v1/                  # REST API routes
│   │           ├── system.py        # /status
│   │           ├── sensors.py       # /readings, /latest, /history
│   │           ├── risk.py          # /current
│   │           ├── scenarios.py     # /scenarios
│   │           ├── alerts.py        # /alerts, /alerts/{id}/acknowledge
│   │           ├── events.py        # /events
│   │           └── camera.py        # /camera/start, /stop, /status, /stream
│   ├── tests/
│   │   ├── conftest.py              # Isolated test database fixture
│   │   ├── test_risk_engine.py      # Risk engine unit tests
│   │   ├── test_api.py              # REST API tests
│   │   ├── test_audit_fixes.py      # Integrity audit tests
│   │   ├── test_persistence.py      # SQLite restart persistence test
│   │   └── test_camera_phase2.py    # Phase 2 camera & streaming unit tests
│   ├── requirements.txt             # Python dependencies (FastAPI, OpenCV, NumPy)
│   └── run.py                       # Backend server launcher
├── frontend/
│   ├── src/
│   │   ├── components/              # Modular UI components
│   │   │   ├── Header.jsx
│   │   │   ├── SystemStatusPanel.jsx
│   │   │   ├── RiskOverview.jsx
│   │   │   ├── SensorCards.jsx
│   │   │   ├── ManualSensorSimulator.jsx
│   │   │   ├── ResearchScenarios.jsx
│   │   │   ├── ContributingFactors.jsx
│   │   │   ├── CameraFeed.jsx        # Phase 2 Live webcam monitoring & controls
│   │   │   ├── ActiveAlerts.jsx
│   │   │   └── DisclaimerFooter.jsx
│   │   ├── services/api.js          # Axios API client (sensors, risk, camera)
│   │   ├── App.jsx                  # Main dashboard layout & telemetry polling
│   │   ├── index.css                # Industrial monitoring stylesheet
│   │   └── main.jsx
│   └── package.json
├── database/
│   └── fire_system.db               # Development/Production SQLite database
├── verify_camera_hardware.py        # Real physical webcam hardware verification script
├── verify_test_db_isolation.py      # Test database isolation hash verification script
└── README.md
```

---

## 4. Phase 2: Real Webcam Integration & OpenCV Video Stream

### What Phase 2 Implements
- **Actual Hardware Acquisition**: Uses `cv2.VideoCapture(camera_index)` to access the local webcam.
- **Controlled Singleton Manager**: [`CameraService`](file:///c:/Users/pc/Documents/research/Horizon%203/fire-detection-system/backend/app/services/camera_service.py) maintains a single hardware capture instance.
- **Dedicated Background Worker**: Non-blocking frame capture loop running on a separate thread, preventing FastAPI requests from blocking.
- **MJPEG Streaming**: Real-time HTTP multipart stream (`multipart/x-mixed-replace; boundary=frame`) directly encoded from live OpenCV frames using `cv2.imencode('.jpg', ...)`.
- **Future AI Architecture Hook**: [`CameraService.get_frame()`](file:///c:/Users/pc/Documents/research/Horizon%203/fire-detection-system/backend/app/services/camera_service.py) provides safe, concurrent frame access for Phase 3 (Person Detector) and Phase 4 (Fire Detector) without opening duplicate video capture handles:
  ```
  Physical Webcam (OpenCV)
            │
            ▼
      CameraService
            │
      Latest Frame
      ├── Frontend Live MJPEG Stream
      ├── Person Detector (Phase 3)
      └── Fire Detector (Phase 4)
  ```
- **Frontend Camera Panel**: Real-time camera monitoring UI featuring **Start Camera** and **Stop Camera** controls, live video viewport, hardware telemetry overlays, and four explicit operational states:
  - `CAMERA DISCONNECTED`: Neutral placeholder; hardware handle released.
  - `CAMERA CONNECTING`: In-progress driver initialization spinner.
  - `CAMERA CONNECTED`: Live MJPEG stream active with pulsing indicator and device metadata.
  - `CAMERA ERROR`: Transparent error notification with troubleshooting tips.
- **Clean Resource Lifecycle**: OpenCV hardware handles are automatically released upon `Stop Camera`, on unrecoverable frame failures, and during FastAPI application lifespan shutdown.
- **Privacy Assurance**: No frames or video feeds are automatically recorded or persisted to disk.

---

## 5. Phase 3: Real YOLO Person Detection & Visible Occupancy Counting

### What Phase 3 Implements
- **Real Pretrained YOLO Model**: Uses the official Ultralytics COCO-pretrained object-detection model `yolo26n.pt` (Ultralytics v8.4.172, PyTorch 2.14.1, CP312).
- **Zero Fake Detections**: No synthetic bounding boxes, no mock counts. Detection is performed strictly against real frames captured by `CameraService`.
- **Single VideoCapture Ownership**: YOLO is **forbidden** from opening the webcam directly (`source=0` is not used). `PersonDetectionService` strictly consumes NumPy frames via `CameraService.get_frame()`.
- **Controlled Background Inference Worker**: Person inference runs in a thread-safe worker loop at a configurable rate (5–10 FPS, default 5.0 FPS) on CPU, preventing inference latency from reducing the physical 30 FPS camera capture or MJPEG streaming responsiveness.
- **Dynamic Class Discovery**: Resolves class ID dynamically where `model.names[cid] == "person"`, ensuring compatibility across model architectures without hardcoded class index assumptions.
- **Presentation Layer Annotation**: Real person bounding boxes, emerald green borders, and confidence labels (`Person {conf}%`) are drawn onto stream frame copies within the presentation layer. The raw frame in `CameraService` remains completely pristine for future Phase 4 fire detection.
- **Stale Detection Protection**: Detections are stamped with generation and timestamp metadata. If camera disconnects, session restarts, or inference is older than 2.0s (`PERSON_STALE_TIMEOUT`), detections are immediately cleared.
- **Decoupled Architecture**: Camera acquisition and Person AI maintain independent states:
  - Camera Connected + Person Model Not Loaded / Error: Valid operational state; camera streams without AI.
  - Model failure does NOT crash FastAPI or stop webcam capture.
- **In-Memory High-Frequency State**: High-frequency detection state is kept strictly in memory (no high-frequency database spam).
- **Risk Assessment Integration**: Recent valid visible person count feeds into the emergency priority determination in `RiskAssessmentService`.

### Centralized Configuration in `app/core/config.py`
```python
PERSON_MODEL_PATH: str = "yolo26n.pt"         # Official Ultralytics COCO detection model
PERSON_CONFIDENCE_THRESHOLD: float = 0.50     # Standard confidence threshold
PERSON_INFERENCE_FPS: float = 5.0             # Controlled background inference rate
PERSON_IMAGE_SIZE: int = 640                  # Input inference resolution
PERSON_DEVICE: str = "cpu"                    # Compute device (cpu / cuda)
PERSON_STALE_TIMEOUT: float = 2.0             # Stale detection timeout (seconds)
```

### Detection REST API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/detection/person/status` | Returns operational status, model loaded flag, enabled state, confidence threshold, visible people count, latency (ms), and effective AI FPS. |
| `GET` | `/api/v1/detection/person/latest` | Returns latest bounding box coordinates, confidence scores, visible count, and stale status. |

---

## 6. Camera Configuration & REST Endpoints

### Centralized Settings in `app/core/config.py`
```python
CAMERA_INDEX: int = 0          # Default video device index
CAMERA_WIDTH: int = 640        # Target capture width
CAMERA_HEIGHT: int = 480       # Target capture height
CAMERA_TARGET_FPS: int = 30    # Target stream frame rate
JPEG_QUALITY: int = 80         # JPEG compression quality (0-100)
```

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/camera/start` | Initializes `VideoCapture` and starts background capture worker. Accepts optional `{"camera_index": 0}`. Returns 503 if unavailable. |
| `POST` | `/api/v1/camera/stop` | Stops worker and releases hardware handle. Idempotent (safe to call multiple times). |
| `GET` | `/api/v1/camera/status` | Returns `{connected, status, camera_index, width, height, fps, error}`. |
| `GET` | `/api/v1/camera/stream` | Streams live multipart MJPEG frames (`multipart/x-mixed-replace; boundary=frame`). |

---

## 6. Camera Troubleshooting & Windows Permissions

### 1. Windows Webcam Permissions
If `Camera unavailable` or `Permission denied` errors occur on Windows:
1. Open **Windows Settings** (`Win + I`).
2. Go to **Privacy & Security** $\to$ **Camera**.
3. Ensure **Camera access** is toggled **On**.
4. Ensure **"Let desktop apps access your camera"** is toggled **On**.

### 2. Device Busy / Locked by Another Application
OpenCV cannot open a webcam if another process has an exclusive lock:
- Close Zoom, Microsoft Teams, Skype, OBS Studio, or browser tabs currently accessing the camera.
- Click **Stop Camera**, wait 1 second, and click **Start Camera** again.

### 3. Multiple Cameras / External USB Webcams
If you have multiple webcams (e.g., integrated laptop webcam + external USB camera):
- Specify the desired index via `POST /api/v1/camera/start` with payload `{"camera_index": 1}`, or set `CAMERA_INDEX=1` in `app/core/config.py`.

---

## 7. Phase 3 Architecture: YOLO Person Detection & Hardened Semantics

### 1. Model & Runtime Specifications
- **Model**: Official Ultralytics COCO-pretrained object detector `yolo26n.pt` (Lightweight Nano variant, 2.4M parameters).
- **Dependency Lock**: Ultralytics pinned to `8.4.172` (`requirements.txt`).
- **Class Filtering**: Strictly filters for the dynamic COCO `person` class ID (ID 0 in standard weights). All non-person classes are filtered out before metric calculation.
- **Hardware Integration**: Runs in a background worker thread (`PersonInferenceWorker`) consuming NumPy frames from `CameraService.get_frame()`. The YOLO worker **never** opens a secondary `cv2.VideoCapture` instance (`source=0` forbidden).

### 2. Person Availability & Semantic Safety
The system strictly distinguishes between valid zero-person detection and detection unavailability:
- **Fresh Valid 0 Occupants**:
  - `people_count: 0`
  - `people_detection_available: true`
  - Emergency priority may downgrade to `HIGH` for confirmed visually clear areas.
- **Unavailable Person Detection**:
  - `people_count: null`
  - `people_detection_available: false`
  - Triggered when: model not loaded, camera disconnected, inference stale (>2.0s), or inference error occurs.
  - **Safety Invariant**: Unknown occupancy **never** reduces emergency priority. `CRITICAL` and `HIGH RISK` conditions retain conservative priorities and explicitly log:  
    *"Person-detection status is currently unavailable; occupancy could not be determined."*  
    Wording such as *"0 people detected"* or *"monitored area clear"* is strictly forbidden when detection is unavailable.
- **Inference Exception Invalidation**:
  - If YOLO inference throws an exception, detection freshness is invalidated immediately (`_last_inference_time = 0.0`), preventing failed frames from masquerading as a zero-person state.
- **Worker Concurrency & Lifecycle Protection**:
  - Synchronized via dedicated `_lifecycle_lock` and generation counter.
  - A replacement `PersonInferenceWorker` cannot start while a previous worker is alive or cleaning up, preventing CPU/GPU resource races.

### 3. Risk Assessment Timing & Sampling Model
> **IMPORTANT ARCHITECTURAL NOTE (Section 8)**:  
> **"The latest fresh person-detection result is sampled when a sensor-based risk assessment is generated."**  
> Therefore, a change in visible-person count alone does not create a new persisted risk assessment until the next sensor assessment event. This design avoids high-frequency database write thrashing at 5–10 FPS while maintaining accurate situational awareness when environmental telemetry arrives. Continuous/event-driven emergency fusion will be addressed in a subsequent phase.

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.12.0)
- Node.js v18+ (tested on Node v22.22.3)
- npm v9+
- Compatible USB or integrated webcam

### 1. Backend Setup & Run
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies (includes FastAPI, OpenCV, NumPy, Ultralytics)
pip install -r requirements.txt

# Run complete automated test suite (68 tests)
pytest -v

# Launch FastAPI backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API is accessible at `http://127.0.0.1:8000`.  
Swagger interactive docs are available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup & Run
```bash
# In a separate terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev -- --host 127.0.0.1 --port 5173
```
Open your browser and navigate to: `http://127.0.0.1:5173/`.

### 3. Physical Hardware Verification Script
To test your physical webcam hardware and YOLO inference independently:
```bash
python verify_phase3_hardware.py
```

---

## 9. Safety & Regulatory Disclaimer

> **IMPORTANT NOTICE**:  
> This system is an academic research prototype developed for testing multi-modal sensor fusion and computer vision algorithms. It is **NOT** a certified life-safety, fire detection, or building evacuation system and must not be used as a primary safety mechanism.
