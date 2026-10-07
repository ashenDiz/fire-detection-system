import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Video,
  VideoOff,
  Play,
  Square,
  AlertTriangle,
  Brain,
  Flame,
  Radio,
  RefreshCw,
  Users,
} from 'lucide-react';
import { cameraApi, detectionApi } from '../services/api';

export default function CameraFeed({ onStatusChange }) {
  // UI states: 'DISCONNECTED', 'CONNECTING', 'CONNECTED', 'ERROR'
  const [uiState, setUiState] = useState('DISCONNECTED');
  const [cameraInfo, setCameraInfo] = useState({
    connected: false,
    status: 'Disconnected',
    camera_index: 0,
    width: 640,
    height: 480,
    fps: 30,
    error: null,
  });
  const [personDetection, setPersonDetection] = useState({
    model_loaded: false,
    status: 'Not Loaded',
    enabled: false,
    people_count: null,
    detection_available: false,
    inference_fps: 0,
    inference_latency_ms: 0,
  });
  const [errorMessage, setErrorMessage] = useState(null);
  const [streamKey, setStreamKey] = useState(() => Date.now());
  const streamImgRef = useRef(null);
  const prevCameraStateRef = useRef(null);
  const intentionalStopRef = useRef(false);

  // Notify parent only when meaningful camera state transitions occur
  const notifyStatusChange = useCallback((newData) => {
    if (!onStatusChange || !newData) return;
    const currentKey = `${Boolean(newData.connected)}_${newData.status || ''}`;
    if (prevCameraStateRef.current === null) {
      prevCameraStateRef.current = currentKey;
      return;
    }
    if (prevCameraStateRef.current !== currentKey) {
      prevCameraStateRef.current = currentKey;
      onStatusChange(newData);
    }
  }, [onStatusChange]);

  // Poll camera status and person detection periodically
  const fetchStatus = useCallback(async () => {
    try {
      const [data, pData] = await Promise.all([
        cameraApi.getStatus(),
        detectionApi.getPersonStatus().catch(() => null),
      ]);
      setCameraInfo(data);
      if (pData) {
        setPersonDetection(pData);
      } else {
        setPersonDetection((prev) => ({
          ...prev,
          people_count: null,
          detection_available: false,
          enabled: false,
          inference_fps: 0,
        }));
      }
      if (data.connected) {
        setUiState('CONNECTED');
        setErrorMessage(null);
      } else {
        if (uiState === 'CONNECTED') {
          // Unexpected disconnection detected by polling (only if not an intentional stop in progress)
          if (!intentionalStopRef.current) {
            setUiState('ERROR');
            setErrorMessage(data.error || 'Camera disconnected unexpectedly.');
          }
        } else if (uiState !== 'CONNECTING' && uiState !== 'ERROR') {
          setUiState('DISCONNECTED');
        }
      }
      notifyStatusChange(data);
    } catch {
      if (uiState === 'CONNECTED' && !intentionalStopRef.current) {
        setUiState('ERROR');
        setErrorMessage('Lost connection to backend camera service.');
        notifyStatusChange({ connected: false, status: 'Error' });
      }
    }
  }, [uiState, notifyStatusChange]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Handle Start Camera
  const handleStart = async () => {
    setUiState('CONNECTING');
    setErrorMessage(null);
    try {
      const result = await cameraApi.start();
      setCameraInfo(result);
      if (result.connected) {
        setUiState('CONNECTED');
        setStreamKey(Date.now());
      } else {
        setUiState('ERROR');
        setErrorMessage(result.error || 'Camera unavailable.');
      }
      notifyStatusChange(result);
    } catch (err) {
      setUiState('ERROR');
      const detail = err.response?.data?.detail || err.message || 'Unable to access webcam.';
      setErrorMessage(detail);
      setCameraInfo((prev) => ({ ...prev, connected: false, status: 'Error' }));
      notifyStatusChange({ connected: false, status: 'Error', error: detail });
    }
  };

  // Handle Stop Camera
  const handleStop = async (options = {}) => {
    const source = options?.source || 'user';
    const preserveError = Boolean(options?.preserveError);
    const errorReason = options?.errorReason || null;

    if (source === 'user') {
      intentionalStopRef.current = true;
    }

    try {
      const result = await cameraApi.stop();
      setCameraInfo(result);
      if (preserveError) {
        setUiState('ERROR');
        if (errorReason) setErrorMessage(errorReason);
      } else {
        setUiState('DISCONNECTED');
        setErrorMessage(null);
      }
      notifyStatusChange(result);
    } catch (err) {
      console.error('Error stopping camera:', err);
      // Stop failure must NOT claim disconnected!
      setUiState('ERROR');
      const failReason = 'Unable to confirm camera shutdown because communication with the backend failed.';
      setErrorMessage(failReason);
      setCameraInfo((prev) => ({ ...prev, connected: false, status: 'Error' }));
      notifyStatusChange({ connected: false, status: 'Error', error: failReason });
    } finally {
      if (source === 'user') {
        intentionalStopRef.current = false;
      }
    }
  };

  // Handle image stream load error
  const handleStreamError = () => {
    // If the user intentionally initiated a Stop, the browser closing the stream is expected
    if (intentionalStopRef.current) {
      return;
    }

    if (uiState === 'CONNECTED') {
      const reason = 'Video stream interrupted or camera hardware is busy.';
      setUiState('ERROR');
      setErrorMessage(reason);
      handleStop({
        source: 'stream-error',
        preserveError: true,
        errorReason: reason,
      });
    }
  };

  return (
    <div className="card" id="camera-module-card">
      <div className="card-header">
        <div className="card-title">
          {uiState === 'CONNECTED' ? (
            <Video size={18} color="#10b981" />
          ) : (
            <VideoOff size={18} color="#94a3b8" />
          )}
          <span>Webcam & Computer Vision Module</span>
        </div>

        {/* State Badge */}
        {uiState === 'CONNECTED' && (
          <span className="badge badge-safe" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span className="pulse-dot" style={{ width: '7px', height: '7px', background: '#10b981', borderRadius: '50%' }} />
            CAMERA CONNECTED
          </span>
        )}
        {uiState === 'CONNECTING' && (
          <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', border: '1px solid #3b82f6' }}>
            <RefreshCw size={12} className="spin" style={{ marginRight: '0.3rem' }} />
            CAMERA CONNECTING
          </span>
        )}
        {uiState === 'DISCONNECTED' && (
          <span className="badge" style={{ background: 'rgba(148, 163, 184, 0.1)', color: '#94a3b8', border: '1px solid #334155' }}>
            CAMERA DISCONNECTED
          </span>
        )}
        {uiState === 'ERROR' && (
          <span className="badge badge-critical" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <AlertTriangle size={12} />
            CAMERA ERROR
          </span>
        )}
      </div>

      {/* Phase 3 Visible Person Occupancy Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.45rem 0.8rem',
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid #1e293b',
          borderRadius: '6px',
          fontSize: '0.82rem',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Users
            size={16}
            color={
              personDetection.model_loaded && uiState === 'CONNECTED' && personDetection.detection_available
                ? (personDetection.people_count > 0 ? '#f59e0b' : '#10b981')
                : '#64748b'
            }
          />
          <span style={{ color: '#cbd5e1', fontWeight: 600 }}>
            {personDetection.model_loaded
              ? (uiState === 'CONNECTED'
                  ? (personDetection.detection_available && personDetection.people_count !== null
                      ? `People Currently Detected by Camera: ${personDetection.people_count}`
                      : 'People Detection: Unavailable')
                  : 'People Detection: Camera Disconnected')
              : 'People Detection: Unavailable'}
          </span>
          {personDetection.model_loaded && uiState === 'CONNECTED' && personDetection.inference_fps > 0 && (
            <span style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'monospace' }}>
              ({personDetection.inference_fps.toFixed(1)} AI FPS &bull; {personDetection.inference_latency_ms.toFixed(0)}ms)
            </span>
          )}
        </div>
        <div style={{ fontSize: '0.72rem', color: '#64748b', fontStyle: 'italic' }}>
          Visible camera FOV only &bull; Not building occupancy
        </div>
      </div>

      {/* Main Video / Placeholder Panel */}
      <div className="camera-video-container" style={{ position: 'relative', width: '100%', minHeight: '260px', background: '#020617', borderRadius: '8px', overflow: 'hidden', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        {uiState === 'CONNECTED' ? (
          <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '260px', display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#000000' }}>
            <img
              ref={streamImgRef}
              key={streamKey}
              src={`${cameraApi.getStreamUrl()}?t=${streamKey}`}
              alt="Live Webcam Stream"
              onError={handleStreamError}
              style={{ width: '100%', maxHeight: '360px', objectFit: 'contain', display: 'block' }}
            />
            {/* Live Indicator Overlay */}
            <div style={{ position: 'absolute', top: '10px', left: '10px', display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(0, 0, 0, 0.75)', padding: '0.25rem 0.6rem', borderRadius: '4px', border: '1px solid rgba(16, 185, 129, 0.4)', fontSize: '0.72rem', fontWeight: 700, color: '#10b981', letterSpacing: '0.05em' }}>
              <Radio size={12} className="pulse" />
              LIVE STREAM
            </div>

            {/* Hardware Telemetry Bar */}
            <div style={{ position: 'absolute', bottom: '0', left: '0', right: '0', background: 'linear-gradient(transparent, rgba(0, 0, 0, 0.85))', padding: '0.4rem 0.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'monospace' }}>
              <span>Camera #{cameraInfo.camera_index} &bull; {cameraInfo.width}x{cameraInfo.height} @ {cameraInfo.fps.toFixed(0)} FPS</span>
              <span>
                {personDetection.model_loaded && personDetection.detection_available && personDetection.people_count !== null
                  ? `People Detected: ${personDetection.people_count}`
                  : 'AI: Unavailable'}
              </span>
            </div>
          </div>
        ) : uiState === 'CONNECTING' ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', padding: '2rem' }}>
            <RefreshCw size={36} color="#3b82f6" className="spin" />
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#93c5fd' }}>
              Initializing OpenCV VideoCapture...
            </div>
            <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
              Accessing hardware camera index {cameraInfo.camera_index}...
            </p>
          </div>
        ) : uiState === 'ERROR' ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', padding: '2rem', textAlign: 'center' }}>
            <AlertTriangle size={36} color="#ef4444" />
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fca5a5' }}>
              Camera Hardware Error
            </div>
            <div style={{ fontSize: '0.8rem', color: '#f87171', maxWidth: '380px', background: 'rgba(239, 68, 68, 0.1)', padding: '0.5rem 0.8rem', borderRadius: '6px', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
              {errorMessage || 'Unable to access webcam. Camera may be in use by another application.'}
            </div>
            <p style={{ fontSize: '0.72rem', color: '#94a3b8', maxWidth: '420px' }}>
              Verify webcam permissions and ensure no other application (e.g. Zoom, Teams, browser) is locking the camera device.
            </p>
          </div>
        ) : (
          /* DISCONNECTED Neutral State */
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', padding: '2rem', textAlign: 'center' }}>
            <VideoOff size={38} color="#64748b" />
            <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#cbd5e1' }}>
              Webcam Hardware Disconnected
            </div>
            <p style={{ fontSize: '0.75rem', maxWidth: '420px', color: '#94a3b8', lineHeight: 1.5 }}>
              Hardware VideoCapture is currently released. Click <strong>Start Camera</strong> to initialize the real OpenCV video stream. No simulated or looped video is displayed.
            </p>
          </div>
        )}
      </div>

      {/* Control Actions & AI State Badges */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', paddingTop: '0.25rem' }}>
        {/* Buttons */}
        <div style={{ display: 'flex', gap: '0.6rem' }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleStart}
            disabled={uiState === 'CONNECTED' || uiState === 'CONNECTING'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: uiState === 'CONNECTED' ? '#334155' : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              borderColor: '#10b981',
              color: '#ffffff',
              fontSize: '0.85rem',
              padding: '0.45rem 0.9rem',
              cursor: uiState === 'CONNECTED' ? 'not-allowed' : 'pointer',
              opacity: uiState === 'CONNECTED' ? 0.6 : 1,
            }}
          >
            <Play size={14} />
            Start Camera
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => handleStop({ source: 'user' })}
            disabled={uiState === 'DISCONNECTED' || uiState === 'CONNECTING'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: uiState === 'DISCONNECTED' ? '#1e293b' : 'rgba(239, 68, 68, 0.15)',
              borderColor: uiState === 'DISCONNECTED' ? '#334155' : 'rgba(239, 68, 68, 0.4)',
              color: uiState === 'DISCONNECTED' ? '#64748b' : '#fca5a5',
              fontSize: '0.85rem',
              padding: '0.45rem 0.9rem',
              cursor: uiState === 'DISCONNECTED' ? 'not-allowed' : 'pointer',
              opacity: uiState === 'DISCONNECTED' ? 0.6 : 1,
            }}
          >
            <Square size={14} />
            Stop Camera
          </button>
        </div>

        {/* Phase Separation Transparency Badges */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span
            className="badge"
            title={personDetection.model_loaded ? 'Ultralytics YOLO COCO Person Detector active' : 'Person model status'}
            style={{
              background: personDetection.status === 'Loaded' ? 'rgba(16, 185, 129, 0.15)' : (personDetection.status.startsWith('Error') ? 'rgba(239, 68, 68, 0.15)' : 'rgba(51, 65, 85, 0.4)'),
              color: personDetection.status === 'Loaded' ? '#34d399' : (personDetection.status.startsWith('Error') ? '#fca5a5' : '#94a3b8'),
              border: personDetection.status === 'Loaded' ? '1px solid rgba(16, 185, 129, 0.3)' : (personDetection.status.startsWith('Error') ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid #334155'),
              fontSize: '0.72rem',
              display: 'inline-flex',
              alignItems: 'center',
              padding: '0.2rem 0.5rem',
              borderRadius: '4px',
            }}
          >
            <Brain size={12} style={{ marginRight: '0.3rem' }} />
            Person Model: {personDetection.status === 'Loaded' ? 'Loaded' : (personDetection.status.startsWith('Error') ? 'Error' : 'Not Loaded')}
          </span>
          <span
            className="badge"
            title="YOLO fire detection scheduled for Phase 4"
            style={{
              background: 'rgba(51, 65, 85, 0.4)',
              color: '#94a3b8',
              border: '1px solid #334155',
              fontSize: '0.72rem',
              display: 'inline-flex',
              alignItems: 'center',
              padding: '0.2rem 0.5rem',
              borderRadius: '4px',
            }}
          >
            <Flame size={12} style={{ marginRight: '0.3rem' }} />
            Fire Model: Not Loaded
          </span>
        </div>
      </div>
    </div>
  );
}
