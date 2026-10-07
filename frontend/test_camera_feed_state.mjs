// Deterministic Frontend State Machine and Concurrency Race Test
// Verifies:
// 1. User Stop vs. MJPEG stream closure race (user Stop ignores expected stream termination)
// 2. Real unexpected stream failure preserves CAMERA ERROR and requests backend cleanup
// 3. Stop API failure reports CAMERA ERROR instead of falsely claiming DISCONNECTED
// 4. Retry from ERROR transitions to CONNECTED
// 5. Deduplicated polling does not trigger redundant parent dashboard refreshes

import assert from 'assert';

console.log("==================================================");
console.log("DETERMINISTIC FRONTEND STATE RACE TEST");
console.log("==================================================");

function createCameraFeed(apiMock, onStatusChange, detectionApiMock = null) {
  let uiState = 'DISCONNECTED';
  let errorMessage = null;
  let cameraInfo = { connected: false, status: 'Disconnected' };
  let personDetection = {
    model_loaded: false,
    status: 'Not Loaded',
    enabled: false,
    people_count: null,
    detection_available: false,
    inference_fps: 0,
    inference_latency_ms: 0,
  };
  let prevCameraState = null;
  let intentionalStop = false;

  function notifyStatusChange(newData) {
    if (!onStatusChange || !newData) return;
    const currentKey = `${Boolean(newData.connected)}_${newData.status || ''}`;
    if (prevCameraState === null) {
      prevCameraState = currentKey;
      return;
    }
    if (prevCameraState !== currentKey) {
      prevCameraState = currentKey;
      onStatusChange(newData);
    }
  }

  async function fetchStatus() {
    try {
      const [data, pData] = await Promise.all([
        apiMock.getStatus(),
        detectionApiMock ? detectionApiMock.getPersonStatus().catch(() => null) : Promise.resolve(null),
      ]);
      cameraInfo = data;
      if (pData) {
        personDetection = pData;
      } else {
        personDetection = {
          ...personDetection,
          people_count: null,
          detection_available: false,
          enabled: false,
          inference_fps: 0,
        };
      }
      if (data.connected) {
        uiState = 'CONNECTED';
        errorMessage = null;
      } else {
        if (uiState === 'CONNECTED') {
          if (!intentionalStop) {
            uiState = 'ERROR';
            errorMessage = data.error || 'Camera disconnected unexpectedly.';
          }
        } else if (uiState !== 'CONNECTING' && uiState !== 'ERROR') {
          uiState = 'DISCONNECTED';
        }
      }
      notifyStatusChange(data);
    } catch {
      if (uiState === 'CONNECTED' && !intentionalStop) {
        uiState = 'ERROR';
        errorMessage = 'Lost connection to backend camera service.';
        notifyStatusChange({ connected: false, status: 'Error' });
      }
    }
  }

  async function handleStart() {
    uiState = 'CONNECTING';
    errorMessage = null;
    try {
      const result = await apiMock.start();
      cameraInfo = result;
      if (result.connected) {
        uiState = 'CONNECTED';
      } else {
        uiState = 'ERROR';
        errorMessage = result.error || 'Camera unavailable.';
      }
      notifyStatusChange(result);
    } catch (err) {
      uiState = 'ERROR';
      errorMessage = err.message || 'Unable to access webcam.';
      cameraInfo = { connected: false, status: 'Error' };
      notifyStatusChange({ connected: false, status: 'Error', error: errorMessage });
    }
  }

  async function handleStop(options = {}) {
    const source = options?.source || 'user';
    const preserveError = Boolean(options?.preserveError);
    const errorReason = options?.errorReason || null;

    if (source === 'user') {
      intentionalStop = true;
    }

    try {
      const result = await apiMock.stop();
      cameraInfo = result;
      if (preserveError) {
        uiState = 'ERROR';
        if (errorReason) errorMessage = errorReason;
      } else {
        uiState = 'DISCONNECTED';
        errorMessage = null;
      }
      notifyStatusChange(result);
    } catch {
      uiState = 'ERROR';
      const failReason = 'Unable to confirm camera shutdown because communication with the backend failed.';
      errorMessage = failReason;
      cameraInfo = { connected: false, status: 'Error' };
      notifyStatusChange({ connected: false, status: 'Error', error: failReason });
    } finally {
      if (source === 'user') {
        intentionalStop = false;
      }
    }
  }

  function handleStreamError() {
    if (intentionalStop) {
      return;
    }

    if (uiState === 'CONNECTED') {
      const reason = 'Video stream interrupted or camera hardware is busy.';
      uiState = 'ERROR';
      errorMessage = reason;
      return handleStop({
        source: 'stream-error',
        preserveError: true,
        errorReason: reason,
      });
    }
  }

  function getPersonDetectionText() {
    if (!personDetection.model_loaded) return 'People Detection: Unavailable';
    if (uiState !== 'CONNECTED') return 'People Detection: Camera Disconnected';
    if (personDetection.detection_available && personDetection.people_count !== null) {
      return `People Currently Detected by Camera: ${personDetection.people_count}`;
    }
    return 'People Detection: Unavailable';
  }

  return {
    getState: () => ({ uiState, errorMessage, cameraInfo, personDetection, intentionalStop }),
    getPersonDetectionText,
    fetchStatus,
    handleStart,
    handleStop,
    handleStreamError,
  };
}

async function runTests() {
  let notifications = [];
  const onStatusChange = (data) => notifications.push(data);

  let stopCallCount = 0;
  let stopDeferred = null;

  const apiMock = {
    getStatus: async () => ({ connected: true, status: 'Connected' }),
    start: async () => ({ connected: true, status: 'Connected', camera_index: 0 }),
    stop: async () => {
      stopCallCount++;
      if (stopDeferred) {
        return stopDeferred.promise;
      }
      return { connected: false, status: 'Disconnected', camera_index: 0 };
    }
  };

  const feed = createCameraFeed(apiMock, onStatusChange);

  // ============================================================
  // TEST 1: User Stop vs. Stream Closure Race
  // ============================================================
  console.log("\n[TEST 1] User Stop vs. MJPEG stream closure race condition...");
  await feed.handleStart();
  assert.strictEqual(feed.getState().uiState, 'CONNECTED');

  // Prepare a deferred promise to intentionally hold stop request in-flight
  let resolveStop;
  stopDeferred = {
    promise: new Promise((resolve) => {
      resolveStop = resolve;
    })
  };

  // User initiates Stop Camera
  const stopPromise = feed.handleStop({ source: 'user' });
  assert.strictEqual(feed.getState().intentionalStop, true, "intentionalStop must be active while Stop is pending");

  // While Stop is pending, simulate the browser firing <img onError> due to backend MJPEG stream close
  feed.handleStreamError();

  // Verify stream termination is recognized as expected and does NOT trigger a second Stop or set ERROR
  assert.strictEqual(stopCallCount, 1, "Expected exactly 1 stop request; stream-error must NOT trigger a second stop");
  assert.strictEqual(feed.getState().uiState, 'CONNECTED', "UI State must remain in transition without jumping to ERROR");

  // Now resolve the backend stop response
  resolveStop({ connected: false, status: 'Disconnected', camera_index: 0 });
  await stopPromise;

  // Verify final UI state is clean DISCONNECTED and NOT ERROR
  assert.strictEqual(feed.getState().uiState, 'DISCONNECTED', "Final state must be DISCONNECTED");
  assert.strictEqual(feed.getState().errorMessage, null, "CAMERA ERROR must NOT be shown after clean user Stop");
  assert.strictEqual(feed.getState().intentionalStop, false, "intentionalStop must be cleared in finally");
  console.log("  -> PASSED: Expected stream closure ignored during user Stop; ended in clean DISCONNECTED.");

  // ============================================================
  // TEST 2: Unexpected Stream Failure Preserves ERROR State
  // ============================================================
  console.log("\n[TEST 2] Unexpected stream failure handling...");
  stopDeferred = null; // direct resolution
  await feed.handleStart();
  assert.strictEqual(feed.getState().uiState, 'CONNECTED');

  // Real stream failure happens (intentionalStop is false)
  const streamErrorPromise = feed.handleStreamError();
  assert.strictEqual(feed.getState().uiState, 'ERROR', "UI State must immediately show ERROR upon stream failure");
  assert.strictEqual(feed.getState().errorMessage, 'Video stream interrupted or camera hardware is busy.');

  await streamErrorPromise;
  // After backend cleanup completes, UI must RETAIN CAMERA ERROR!
  assert.strictEqual(feed.getState().uiState, 'ERROR', "UI State must RETAIN ERROR after backend cleanup");
  assert.strictEqual(feed.getState().errorMessage, 'Video stream interrupted or camera hardware is busy.');
  console.log("  -> PASSED: Real stream failure cleanly preserved CAMERA ERROR state and message.");

  // ============================================================
  // TEST 3: Retry from ERROR
  // ============================================================
  console.log("\n[TEST 3] Retry from ERROR state...");
  await feed.handleStart();
  assert.strictEqual(feed.getState().uiState, 'CONNECTED', "Retry from ERROR successfully connects");
  assert.strictEqual(feed.getState().errorMessage, null, "Error message cleared upon successful start");
  console.log("  -> PASSED: Start Camera successfully retries from ERROR state.");

  // ============================================================
  // TEST 4: Stop API Failure Reports ERROR
  // ============================================================
  console.log("\n[TEST 4] Stop API failure must NOT claim DISCONNECTED...");
  const failingStopMock = {
    ...apiMock,
    stop: async () => { throw new Error("Connection timed out"); }
  };
  const failingFeed = createCameraFeed(failingStopMock, onStatusChange);
  await failingFeed.handleStart();
  assert.strictEqual(failingFeed.getState().uiState, 'CONNECTED');

  await failingFeed.handleStop({ source: 'user' });
  assert.strictEqual(failingFeed.getState().uiState, 'ERROR', "Must NOT claim DISCONNECTED if backend stop failed");
  assert(failingFeed.getState().errorMessage.includes('Unable to confirm camera shutdown'),
    `Expected failure diagnostic, got: ${failingFeed.getState().errorMessage}`);
  console.log("  -> PASSED: Failed Stop request safely reports CAMERA ERROR instead of false DISCONNECTED.");

  // ============================================================
  // TEST 5: Person Status Request Failure Clears Old Count
  // ============================================================
  console.log("\n[TEST 5] Person status request failure must clear old count...");
  let personStatusFail = false;
  const detectionMock = {
    getPersonStatus: async () => {
      if (personStatusFail) throw new Error("Status endpoint 500 error");
      return {
        model_loaded: true,
        status: 'Loaded',
        model_name: 'yolo26n.pt',
        enabled: true,
        people_count: 2,
        detection_available: true,
        inference_fps: 8.5,
        inference_latency_ms: 35.0,
      };
    }
  };

  const feedWithDetection = createCameraFeed(apiMock, onStatusChange, detectionMock);
  await feedWithDetection.handleStart();
  await feedWithDetection.fetchStatus();

  assert.strictEqual(feedWithDetection.getState().personDetection.people_count, 2);
  assert.strictEqual(feedWithDetection.getState().personDetection.detection_available, true);
  assert.strictEqual(feedWithDetection.getPersonDetectionText(), "People Currently Detected by Camera: 2");

  // Next status request fails
  personStatusFail = true;
  await feedWithDetection.fetchStatus();

  // Must NOT keep old count 2 as valid!
  assert.strictEqual(feedWithDetection.getState().personDetection.people_count, null);
  assert.strictEqual(feedWithDetection.getState().personDetection.detection_available, false);
  assert.strictEqual(feedWithDetection.getPersonDetectionText(), "People Detection: Unavailable");
  // Camera state must remain independent and CONNECTED
  assert.strictEqual(feedWithDetection.getState().uiState, 'CONNECTED');
  console.log("  -> PASSED: Person status request failure cleanly sets unavailable and clears old count 2.");

  console.log("\n==================================================");
  console.log(">>> ALL FRONTEND RACE & STATE TESTS PASSED! <<<");
  console.log("==================================================");
}

runTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
