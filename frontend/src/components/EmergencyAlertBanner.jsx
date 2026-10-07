import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AlertTriangle,
  Flame,
  Volume2,
  VolumeX,
  ShieldAlert,
  Users,
  Radio,
} from 'lucide-react';

import { emergencyApi } from '../services/api';

const RAW_FIRE_DETECTION_THRESHOLD = 0.30;
const POLL_INTERVAL_MS = 750;

export default function EmergencyAlertBanner() {
  const [emergencyStatus, setEmergencyStatus] = useState(null);
  const [apiError, setApiError] = useState(false);
  const [audioArmed, setAudioArmed] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const audioCtxRef = useRef(null);
  const alarmIntervalRef = useRef(null);
  const activeOscillatorsRef = useRef(new Set());

  // Polling lifecycle / overlap protection
  const pollTimerRef = useRef(null);
  const pollInFlightRef = useRef(false);

  // ---------------------------------------------------------
  // Emergency API Polling
  // ---------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    const scheduleNextPoll = () => {
      if (cancelled) return;

      pollTimerRef.current = setTimeout(() => {
        runPoll();
      }, POLL_INTERVAL_MS);
    };

    const runPoll = async () => {
      if (cancelled) return;

      // Never allow overlapping emergency-status requests.
      if (pollInFlightRef.current) {
        scheduleNextPoll();
        return;
      }

      pollInFlightRef.current = true;

      try {
        const data = await emergencyApi.getStatus();

        if (!cancelled) {
          setEmergencyStatus(data);
          setApiError(false);
        }
      } catch (err) {
        console.warn('Failed to poll emergency status:', err);

        if (!cancelled) {
          // IMPORTANT:
          // Keep the last known emergencyStatus.
          // API failure must never fabricate NORMAL or clear an emergency.
          setApiError(true);
        }
      } finally {
        pollInFlightRef.current = false;
        scheduleNextPoll();
      }
    };

    runPoll();

    return () => {
      cancelled = true;

      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, []);

  // ---------------------------------------------------------
  // Web Audio Alarm
  // ---------------------------------------------------------

  const stopAlarmSound = useCallback(() => {
    if (alarmIntervalRef.current) {
      clearInterval(alarmIntervalRef.current);
      alarmIntervalRef.current = null;
    }

    // Immediately stop any pulse that is currently sounding.
    activeOscillatorsRef.current.forEach((oscillator) => {
      try {
        oscillator.stop();
      } catch {
        // Oscillator may already have stopped.
      }

      try {
        oscillator.disconnect();
      } catch {
        // Ignore cleanup failures.
      }
    });

    activeOscillatorsRef.current.clear();
  }, []);

  const startAlarmSound = useCallback(() => {
    if (!audioArmed || isMuted) {
      stopAlarmSound();
      return;
    }

    try {
      const AudioCtxClass =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioCtxClass) {
        console.warn('Web Audio API is not supported by this browser.');
        return;
      }

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioCtxClass();
      }

      const ctx = audioCtxRef.current;

      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => { });
      }

      // Prevent duplicate alarm intervals.
      if (alarmIntervalRef.current) {
        return;
      }

      let highPitch = true;

      const playPulse = () => {
        try {
          if (ctx.state === 'suspended') {
            ctx.resume().catch(() => { });
          }

          const oscillator = ctx.createOscillator();
          const gain = ctx.createGain();

          oscillator.type = 'sawtooth';
          oscillator.frequency.setValueAtTime(
            highPitch ? 880 : 660,
            ctx.currentTime
          );

          highPitch = !highPitch;

          gain.gain.setValueAtTime(0.25, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(
            0.01,
            ctx.currentTime + 0.28
          );

          oscillator.connect(gain);
          gain.connect(ctx.destination);

          activeOscillatorsRef.current.add(oscillator);

          oscillator.onended = () => {
            activeOscillatorsRef.current.delete(oscillator);

            try {
              oscillator.disconnect();
            } catch {
              // Ignore cleanup errors.
            }

            try {
              gain.disconnect();
            } catch {
              // Ignore cleanup errors.
            }
          };

          oscillator.start(ctx.currentTime);
          oscillator.stop(ctx.currentTime + 0.3);
        } catch (err) {
          console.debug('Audio pulse error:', err);
        }
      };

      playPulse();

      alarmIntervalRef.current = setInterval(
        playPulse,
        400
      );
    } catch (err) {
      console.warn('Web Audio synthesis error:', err);
    }
  }, [audioArmed, isMuted, stopAlarmSound]);

  // Alarm is controlled ONLY by the last known backend emergency state.
  useEffect(() => {
    const isConfirmed =
      emergencyStatus?.state === 'EMERGENCY_CONFIRMED';

    if (isConfirmed && audioArmed && !isMuted) {
      startAlarmSound();
    } else {
      stopAlarmSound();
    }

    return () => {
      stopAlarmSound();
    };
  }, [
    emergencyStatus?.state,
    audioArmed,
    isMuted,
    startAlarmSound,
    stopAlarmSound,
  ]);

  // Full audio cleanup on unmount.
  useEffect(() => {
    return () => {
      stopAlarmSound();

      if (
        audioCtxRef.current &&
        audioCtxRef.current.state !== 'closed'
      ) {
        audioCtxRef.current.close().catch(() => { });
      }

      audioCtxRef.current = null;
    };
  }, [stopAlarmSound]);

  // ---------------------------------------------------------
  // Audio Controls
  // ---------------------------------------------------------

  const handleArmAudio = useCallback(() => {
    try {
      const AudioCtxClass =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioCtxClass) {
        console.warn('Web Audio API is not supported by this browser.');
        return;
      }

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioCtxClass();
      }

      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume().catch(() => { });
      }

      setAudioArmed(true);
    } catch (err) {
      console.warn('Could not arm audio context:', err);
    }
  }, []);

  // Automatically arm emergency audio on the user's first interaction
  // anywhere in the dashboard.
  //
  // Browsers require one user gesture before audible Web Audio is allowed.
  // The user does NOT need to press a dedicated "Enable Alarm" button.
  useEffect(() => {
    if (audioArmed) return;

    const autoArmAudio = () => {
      handleArmAudio();
    };

    window.addEventListener('pointerdown', autoArmAudio, {
      once: true,
      capture: true,
    });

    window.addEventListener('keydown', autoArmAudio, {
      once: true,
      capture: true,
    });

    return () => {
      window.removeEventListener(
        'pointerdown',
        autoArmAudio,
        true
      );

      window.removeEventListener(
        'keydown',
        autoArmAudio,
        true
      );
    };
  }, [audioArmed, handleArmAudio]);

  const handleToggleMute = () => {
    setIsMuted((prev) => !prev);
  };

  const renderAlarmControls = (compact = false) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.6rem',
        flexWrap: 'wrap',
      }}
    >
      {!audioArmed ? (
        <span
          style={{
            fontSize: '0.8rem',
            fontWeight: 700,
            opacity: 0.9,
          }}
        >
          🔊 Emergency Alarm: Automatic
        </span>
      ) : (
        <>
          {!compact && (
            <span
              style={{
                fontSize: '0.8rem',
                fontWeight: 700,
                opacity: 0.9,
              }}
            >
              Emergency Alarm:{' '}
              {isMuted ? 'Muted' : 'Automatic'}
            </span>
          )}

          <button
            id="btn-toggle-mute"
            type="button"
            onClick={handleToggleMute}
            style={{
              background: isMuted
                ? 'rgba(0, 0, 0, 0.35)'
                : '#ffffff',
              color: isMuted ? '#ffffff' : '#b91c1c',
              border: isMuted
                ? '1px solid rgba(255,255,255,0.4)'
                : 'none',
              borderRadius: '6px',
              padding: compact
                ? '0.45rem 0.7rem'
                : '0.55rem 0.9rem',
              fontWeight: 700,
              fontSize: '0.82rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              cursor: 'pointer',
            }}
          >
            {isMuted ? (
              <VolumeX size={16} />
            ) : (
              <Volume2 size={16} />
            )}

            {isMuted ? 'Unmute Alarm' : 'Mute Alarm'}
          </button>
        </>
      )}
    </div>
  );

  // ---------------------------------------------------------
  // NORMAL / INITIAL STATE
  // ---------------------------------------------------------

  if (
    !emergencyStatus ||
    emergencyStatus.state === 'NORMAL'
  ) {
    return (
      <div
        id={
          apiError
            ? 'emergency-banner-degraded'
            : 'emergency-alarm-control'
        }
        style={{
          background: apiError
            ? 'rgba(239, 68, 68, 0.12)'
            : 'rgba(15, 23, 42, 0.55)',
          border: apiError
            ? '1px solid rgba(239, 68, 68, 0.45)'
            : '1px solid rgba(148, 163, 184, 0.18)',
          borderRadius: '8px',
          padding: '0.65rem 1rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          flexWrap: 'wrap',
          color: apiError ? '#fca5a5' : '#cbd5e1',
          fontSize: '0.82rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
          }}
        >
          {apiError ? (
            <Radio size={18} />
          ) : (
            <ShieldAlert size={18} />
          )}

          <div>
            <div
              style={{
                fontWeight: 700,
              }}
            >
              Emergency Alarm:{' '}
              {isMuted
                ? 'Muted'
                : 'Automatic'}
            </div>

            <div
              style={{
                marginTop: '0.15rem',
                opacity: 0.8,
              }}
            >
              {apiError
                ? 'Emergency response status is currently unavailable. No emergency state is being inferred from this connection failure.'
                : emergencyStatus
                  ? 'No emergency is currently confirmed.'
                  : 'Preparing emergency response monitoring...'}
            </div>
          </div>
        </div>

        {renderAlarmControls(true)}
      </div>
    );
  }

  // ---------------------------------------------------------
  // MONITORING STATE
  // ---------------------------------------------------------

  if (emergencyStatus.state === 'MONITORING') {
    const confPct =
      emergencyStatus.fire_confidence != null
        ? (
          emergencyStatus.fire_confidence * 100
        ).toFixed(1)
        : 'N/A';

    const confirmationPct =
      emergencyStatus.confirmation_threshold != null
        ? (
          emergencyStatus.confirmation_threshold * 100
        ).toFixed(0)
        : 'N/A';

    return (
      <div
        id="emergency-banner-monitoring"
        style={{
          background:
            'linear-gradient(90deg, rgba(217, 119, 6, 0.25) 0%, rgba(180, 83, 9, 0.2) 100%)',
          border: '1px solid #d97706',
          borderRadius: '8px',
          padding: '0.85rem 1.25rem',
          marginBottom: '1.25rem',
          color: '#fbbf24',
          boxShadow:
            '0 4px 12px rgba(217, 119, 6, 0.15)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.85rem',
            }}
          >
            <AlertTriangle
              size={24}
              color="#f59e0b"
              style={{ flexShrink: 0 }}
            />

            <div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: '0.95rem',
                  letterSpacing: '0.02em',
                  color: '#fef3c7',
                }}
              >
                POSSIBLE VISUAL FIRE DETECTED —
                MONITORING FOR CONFIRMATION
              </div>

              <div
                style={{
                  fontSize: '0.82rem',
                  color: '#fde68a',
                  marginTop: '0.2rem',
                }}
              >
                Candidate Confidence:{' '}
                <strong>{confPct}%</strong>
                {' — '}
                Raw Detection Threshold:{' '}
                {(
                  RAW_FIRE_DETECTION_THRESHOLD *
                  100
                ).toFixed(0)}
                %
                {' | '}
                Required for Confirmation:{' '}
                {confirmationPct}%
              </div>

              <div
                style={{
                  fontSize: '0.8rem',
                  color: '#fde68a',
                  marginTop: '0.25rem',
                }}
              >
                {emergencyStatus.reason}
              </div>
            </div>
          </div>

          <div
            style={{
              display: 'flex',
              gap: '0.6rem',
              alignItems: 'center',
              flexWrap: 'wrap',
            }}
          >
            {renderAlarmControls(true)}

            <div
              style={{
                fontSize: '0.78rem',
                color: '#fbbf24',
                background: 'rgba(0,0,0,0.3)',
                padding: '0.35rem 0.65rem',
                borderRadius: '4px',
                fontWeight: 600,
              }}
            >
              STATUS: MONITORING
            </div>
          </div>
        </div>

        {apiError && (
          <div
            style={{
              marginTop: '0.75rem',
              padding: '0.55rem 0.75rem',
              borderRadius: '6px',
              background: 'rgba(127, 29, 29, 0.35)',
              border:
                '1px solid rgba(248, 113, 113, 0.45)',
              color: '#fecaca',
              fontSize: '0.8rem',
              fontWeight: 600,
            }}
          >
            ⚠ Emergency status connection lost —
            showing the last known MONITORING state.
            No emergency escalation is being inferred
            from the communication failure.
          </div>
        )}
      </div>
    );
  }

  // ---------------------------------------------------------
  // EMERGENCY CONFIRMED
  // ---------------------------------------------------------

  if (
    emergencyStatus.state ===
    'EMERGENCY_CONFIRMED'
  ) {
    const isSensorOnly =
      emergencyStatus.confirmation_source ===
      'SENSOR_CRITICAL';

    const emergencyTitle = isSensorOnly
      ? '🚨 CRITICAL FIRE-RISK EMERGENCY'
      : '🚨 FIRE EMERGENCY CONFIRMED';

    const fireEvidence =
      emergencyStatus.fire_confidence != null
        ? `${(
          emergencyStatus.fire_confidence * 100
        ).toFixed(1)}%`
        : isSensorOnly
          ? 'Manual simulated environmental telemetry'
          : 'Not Available';

    let occupancyText =
      'Occupancy: Person-detection status is currently unavailable; occupancy could not be verified.';

    if (
      emergencyStatus.people_detection_available &&
      emergencyStatus.visible_people_count !== null
    ) {
      if (
        emergencyStatus.visible_people_count === 0
      ) {
        occupancyText =
          'Occupancy: 0 people currently detected by camera. Actual occupancy is not confirmed.';
      } else {
        occupancyText = `Occupancy: ${emergencyStatus.visible_people_count} person(s) currently detected in camera view.`;
      }
    }

    return (
      <div
        id="emergency-banner-confirmed"
        style={{
          background:
            'linear-gradient(135deg, rgba(220, 38, 38, 0.95) 0%, rgba(153, 27, 27, 0.95) 100%)',
          border: '2px solid #ef4444',
          borderRadius: '10px',
          padding: '1.1rem 1.5rem',
          marginBottom: '1.25rem',
          color: '#ffffff',
          boxShadow:
            '0 0 24px rgba(239, 68, 68, 0.5), 0 4px 12px rgba(0, 0, 0, 0.4)',
          animation: 'pulse 2s infinite',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '1rem',
            }}
          >
            <div
              style={{
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '0.6rem',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Flame
                size={32}
                color="#ffffff"
              />
            </div>

            <div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  flexWrap: 'wrap',
                }}
              >
                <span
                  style={{
                    fontSize: '1.2rem',
                    fontWeight: 900,
                    letterSpacing: '0.05em',
                    textTransform: 'uppercase',
                  }}
                >
                  {emergencyTitle}
                </span>

                <span
                  style={{
                    background: '#ffffff',
                    color: '#b91c1c',
                    padding: '0.2rem 0.55rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 800,
                  }}
                >
                  {emergencyStatus.priority} PRIORITY
                </span>
              </div>

              <div
                style={{
                  marginTop: '0.35rem',
                  fontSize: '0.92rem',
                  opacity: 0.95,
                  fontWeight: 500,
                }}
              >
                {emergencyStatus.reason}
              </div>

              <div
                style={{
                  display: 'flex',
                  gap: '1.25rem',
                  marginTop: '0.5rem',
                  fontSize: '0.82rem',
                  opacity: 0.9,
                  flexWrap: 'wrap',
                }}
              >
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                  }}
                >
                  <ShieldAlert size={15} />
                  Fire Evidence:{' '}
                  <strong>{fireEvidence}</strong>
                </span>

                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                  }}
                >
                  <Users size={15} />
                  {occupancyText}
                </span>
              </div>

              {emergencyStatus.confirmation_source && (
                <div
                  style={{
                    marginTop: '0.45rem',
                    fontSize: '0.76rem',
                    opacity: 0.82,
                  }}
                >
                  Confirmation Source:{' '}
                  {emergencyStatus.confirmation_source}
                </div>
              )}
            </div>
          </div>

          {renderAlarmControls(false)}
        </div>

        {apiError && (
          <div
            id="emergency-connection-warning"
            style={{
              marginTop: '0.9rem',
              padding: '0.65rem 0.8rem',
              borderRadius: '6px',
              background: 'rgba(0, 0, 0, 0.35)',
              border:
                '1px solid rgba(254, 240, 138, 0.65)',
              color: '#fef08a',
              fontSize: '0.82rem',
              fontWeight: 700,
            }}
          >
            ⚠ Emergency status connection lost.
            The last confirmed emergency state remains
            active until a safe backend update is
            received.
          </div>
        )}
      </div>
    );
  }

  // Unknown backend state — do not fabricate safety or emergency.
  return (
    <div
      style={{
        background: 'rgba(239, 68, 68, 0.12)',
        border:
          '1px solid rgba(239, 68, 68, 0.4)',
        borderRadius: '8px',
        padding: '0.75rem 1rem',
        marginBottom: '1rem',
        color: '#fca5a5',
      }}
    >
      Emergency response state is currently
      unavailable or unsupported.
    </div>
  );
}