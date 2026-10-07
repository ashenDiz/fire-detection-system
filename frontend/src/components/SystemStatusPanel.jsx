import React from 'react';
import { Database, Camera, Brain, Activity, Sliders, CheckCircle2, AlertCircle } from 'lucide-react';

export default function SystemStatusPanel({ status }) {
  const isDbOk = status?.database === 'Connected';
  const isCamOk = status?.camera === 'Connected';
  const isPersonOk = status?.person_model?.includes('Loaded');
  const isFireOk = status?.fire_model?.includes('Model Loaded');

  return (
    <div className="status-bar" id="system-status-panel">
      <div className="status-item">
        <span className="status-label">Backend API</span>
        <span className="status-value" style={{ color: status?.backend === 'Online' ? '#34d399' : '#f87171' }}>
          <Activity size={15} />
          {status?.backend || 'Checking...'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">Database</span>
        <span className="status-value" style={{ color: isDbOk ? '#34d399' : '#f87171' }}>
          <Database size={15} />
          {status?.database || 'Connecting...'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">Camera</span>
        <span className="status-value" style={{ color: isCamOk ? '#34d399' : '#94a3b8' }}>
          <Camera size={15} />
          {status?.camera || 'Disconnected'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">Person AI Model</span>
        <span className="status-value" style={{ color: isPersonOk ? '#34d399' : '#94a3b8' }}>
          <Brain size={15} />
          {status?.person_model || 'Not Loaded (Phase 1)'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">Fire Detection Model</span>
        <span className="status-value" style={{ color: isFireOk ? '#34d399' : '#f59e0b' }}>
          <AlertCircle size={15} />
          {status?.fire_model || 'Fire AI Model Not Loaded'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">Sensor Telemetry</span>
        <span className="status-value" style={{ color: '#38bdf8' }}>
          <Sliders size={15} />
          {status?.sensor_mode || 'Manual Simulation'}
        </span>
      </div>
    </div>
  );
}
