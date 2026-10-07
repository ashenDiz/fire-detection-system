import assert from 'assert';

console.log(
  '=================================================='
);
console.log(
  'PHASE 5B FRONTEND EMERGENCY BANNER & ALARM TEST'
);
console.log(
  '=================================================='
);

function createEmergencyBannerState(apiMock) {
  let emergencyStatus = null;
  let apiError = false;
  let audioArmed = false;
  let isMuted = false;
  let soundPlaying = false;
  let pollInFlight = false;

  function updateAudioState() {
    const isConfirmed =
      emergencyStatus?.state ===
      'EMERGENCY_CONFIRMED';

    if (
      isConfirmed &&
      audioArmed &&
      !isMuted
    ) {
      soundPlaying = true;
    } else {
      soundPlaying = false;
    }
  }

  async function poll() {
    // Simulates production overlap protection.
    if (pollInFlight) {
      return false;
    }

    pollInFlight = true;

    try {
      const data =
        await apiMock.getStatus();

      emergencyStatus = data;
      apiError = false;
    } catch {
      // IMPORTANT:
      // Preserve last-known status.
      apiError = true;
    } finally {
      pollInFlight = false;
    }

    updateAudioState();

    return true;
  }

  function armAudio() {
    audioArmed = true;
    isMuted = false;
    updateAudioState();
  }

  function toggleMute() {
    isMuted = !isMuted;
    updateAudioState();
  }

  function getRenderView() {
    if (!emergencyStatus) {
      return apiError
        ? 'DEGRADED_WITH_ARM_CONTROL'
        : 'ALARM_ARM_CONTROL';
    }

    if (
      emergencyStatus.state ===
      'EMERGENCY_CONFIRMED'
    ) {
      return apiError
        ? 'RED_CONFIRMED_BANNER_DEGRADED'
        : 'RED_CONFIRMED_BANNER';
    }

    if (
      emergencyStatus.state === 'MONITORING'
    ) {
      return apiError
        ? 'AMBER_MONITORING_BANNER_DEGRADED'
        : 'AMBER_MONITORING_BANNER';
    }

    if (
      emergencyStatus.state === 'NORMAL'
    ) {
      return apiError
        ? 'DEGRADED_WITH_ARM_CONTROL'
        : 'NORMAL_ARM_CONTROL';
    }

    return 'UNKNOWN';
  }

  return {
    poll,
    armAudio,
    toggleMute,
    getRenderView,
    getEmergencyStatus: () =>
      emergencyStatus,
    isApiError: () => apiError,
    isSoundPlaying: () =>
      soundPlaying,
    isMuted: () => isMuted,
    isArmed: () => audioArmed,
    isPollInFlight: () =>
      pollInFlight,
  };
}

async function runTests() {
  let mockStatusResponse = {
    state: 'NORMAL',
    priority: 'LOW',
    reason:
      'Normal operating conditions',
    fire_confidence: 0.0,
    visible_people_count: 0,
    people_detection_available: true,
  };

  let failApi = false;

  const mockApi = {
    getStatus: async () => {
      if (failApi) {
        throw new Error(
          'Network error'
        );
      }

      return {
        ...mockStatusResponse,
      };
    },
  };

  const state =
    createEmergencyBannerState(
      mockApi
    );

  // --------------------------------------------------
  // TEST 1
  // NORMAL exposes alarm control
  // --------------------------------------------------

  console.log(
    '[TEST 1] NORMAL alarm-arm control...'
  );

  await state.poll();

  assert.strictEqual(
    state.getRenderView(),
    'NORMAL_ARM_CONTROL'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  assert.strictEqual(
    state.isArmed(),
    false
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 2
  // Arm sound during NORMAL
  // --------------------------------------------------

  console.log(
    '[TEST 2] Arm alarm while NORMAL...'
  );

  state.armAudio();

  assert.strictEqual(
    state.isArmed(),
    true
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 3
  // NORMAL -> MONITORING
  // --------------------------------------------------

  console.log(
    '[TEST 3] MONITORING state...'
  );

  mockStatusResponse = {
    state: 'MONITORING',
    priority: 'MEDIUM',
    reason:
      'Possible visual fire detected — monitoring for confirmation',
    fire_confidence: 0.35,
    visible_people_count: 0,
    people_detection_available: true,
  };

  await state.poll();

  assert.strictEqual(
    state.getRenderView(),
    'AMBER_MONITORING_BANNER'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 4
  // Armed NORMAL -> confirmed emergency
  // --------------------------------------------------

  console.log(
    '[TEST 4] Armed emergency transition...'
  );

  mockStatusResponse = {
    state: 'EMERGENCY_CONFIRMED',
    priority: 'CRITICAL',
    reason:
      'Visual fire emergency confirmed',
    fire_confidence: 0.72,
    visible_people_count: 2,
    people_detection_available: true,
  };

  await state.poll();

  assert.strictEqual(
    state.getRenderView(),
    'RED_CONFIRMED_BANNER'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    true,
    'Alarm must start automatically because audio was armed earlier'
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 5
  // Mute
  // --------------------------------------------------

  console.log(
    '[TEST 5] Mute alarm...'
  );

  state.toggleMute();

  assert.strictEqual(
    state.isMuted(),
    true
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  assert.strictEqual(
    state.getRenderView(),
    'RED_CONFIRMED_BANNER'
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 6
  // Unmute
  // --------------------------------------------------

  console.log(
    '[TEST 6] Unmute alarm...'
  );

  state.toggleMute();

  assert.strictEqual(
    state.isMuted(),
    false
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    true
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 7
  // API failure during confirmed emergency
  // --------------------------------------------------

  console.log(
    '[TEST 7] API failure during confirmed emergency...'
  );

  failApi = true;

  await state.poll();

  assert.strictEqual(
    state.isApiError(),
    true
  );

  assert.strictEqual(
    state.getRenderView(),
    'RED_CONFIRMED_BANNER_DEGRADED'
  );

  assert.strictEqual(
    state.getEmergencyStatus()
      .state,
    'EMERGENCY_CONFIRMED'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    true,
    'Alarm must remain active during communication failure'
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 8
  // Backend recovers and returns NORMAL
  // --------------------------------------------------

  console.log(
    '[TEST 8] Safe backend recovery...'
  );

  failApi = false;

  mockStatusResponse = {
    state: 'NORMAL',
    priority: 'LOW',
    reason:
      'Previous emergency cleared',
    fire_confidence: 0.0,
    visible_people_count: 0,
    people_detection_available: true,
  };

  await state.poll();

  assert.strictEqual(
    state.isApiError(),
    false
  );

  assert.strictEqual(
    state.getRenderView(),
    'NORMAL_ARM_CONTROL'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false,
    'Alarm must stop after real backend NORMAL response'
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 9
  // API failure while NORMAL
  // --------------------------------------------------

  console.log(
    '[TEST 9] API failure while NORMAL...'
  );

  failApi = true;

  await state.poll();

  assert.strictEqual(
    state.getRenderView(),
    'DEGRADED_WITH_ARM_CONTROL'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  assert.strictEqual(
    state.getEmergencyStatus()
      .state,
    'NORMAL'
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 10
  // API failure while monitoring
  // --------------------------------------------------

  console.log(
    '[TEST 10] API failure while MONITORING...'
  );

  failApi = false;

  mockStatusResponse = {
    state: 'MONITORING',
    priority: 'MEDIUM',
    reason:
      'Possible visual fire detected',
    fire_confidence: 0.36,
    visible_people_count: 0,
    people_detection_available: true,
  };

  await state.poll();

  failApi = true;

  await state.poll();

  assert.strictEqual(
    state.getRenderView(),
    'AMBER_MONITORING_BANNER_DEGRADED'
  );

  assert.strictEqual(
    state.getEmergencyStatus()
      .state,
    'MONITORING'
  );

  assert.strictEqual(
    state.isSoundPlaying(),
    false
  );

  console.log(
    '  -> PASSED'
  );

  // --------------------------------------------------
  // TEST 11
  // Poll overlap protection
  // --------------------------------------------------

  console.log(
    '[TEST 11] Prevent overlapping API polls...'
  );

  let resolveRequest;
  let requestCount = 0;

  const slowApi = {
    getStatus: () => {
      requestCount += 1;

      return new Promise(
        (resolve) => {
          resolveRequest = resolve;
        }
      );
    },
  };

  const slowState =
    createEmergencyBannerState(
      slowApi
    );

  const firstPoll =
    slowState.poll();

  assert.strictEqual(
    slowState.isPollInFlight(),
    true
  );

  const secondPollResult =
    await slowState.poll();

  assert.strictEqual(
    secondPollResult,
    false,
    'Second overlapping poll must be rejected'
  );

  assert.strictEqual(
    requestCount,
    1,
    'Only one API request may be in flight'
  );

  resolveRequest({
    state: 'NORMAL',
    priority: 'LOW',
    reason: 'Normal',
  });

  await firstPoll;

  assert.strictEqual(
    slowState.isPollInFlight(),
    false
  );

  assert.strictEqual(
    requestCount,
    1
  );

  console.log(
    '  -> PASSED'
  );

  console.log(
    '=================================================='
  );

  console.log(
    '>>> ALL FRONTEND EMERGENCY STATE TESTS PASSED! <<<'
  );

  console.log(
    '=================================================='
  );
}

runTests().catch((err) => {
  console.error(
    'Test failed:',
    err
  );

  process.exit(1);
});