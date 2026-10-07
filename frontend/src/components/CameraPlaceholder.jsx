import React from 'react';
import { VideoOff, Brain, AlertTriangle } from 'lucide-react';

export default function CameraPlaceholder({ status }) {
  return (
    <div className="card" id="camera-module-card">
      <div className="card-header">
        <div className="card-title">
          <VideoOff size={18} color="#94a3b8" />
          Webcam & Computer Vision Module
        </div>
        <span className="badge" style={{ background: 'rgba(148, 163, 184, 0.1)', color: '#94a3b8', border: '1px solid #334155' }}>
          Phase 1: Inactive
        </span>
      </div>

      <div className="camera-preview-panel">
        <VideoOff size={38} color="#64748b" />
        <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#cbd5e1' }}>
          Camera Hardware Disconnected
        </div>
        <p style={{ fontSize: '0.75rem', maxWidth: '420px', color: '#94a3b8' }}>
          In accordance with research transparency standards, no simulated or looped video stream is rendered. Real-time OpenCV video streaming and frame capture are scheduled for <strong>Phase 2</strong>.
        </p>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          <span className="badge badge-caution">
            <Brain size={13} />
            Person Model: Not Loaded
          </span>
          <span className="badge badge-warning">
            <AlertTriangle size={13} />
            Fire AI Model Not Loaded
          </span>
        </div>
      </div>
    </div>
  );
}
