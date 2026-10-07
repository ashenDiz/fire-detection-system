import React from 'react';
import { Thermometer, Droplets, CloudFog, Gauge, Flame } from 'lucide-react';

export default function SensorCards({ sensorData }) {
  const temp = sensorData?.temperature ?? 0;
  const humidity = sensorData?.humidity ?? 0;
  const smoke = sensorData?.smoke_level ?? 0;
  const gas = sensorData?.gas_level ?? 0;
  const flame = sensorData?.flame_level ?? 0;

  return (
    <div className="card" id="sensor-telemetry-card">
      <div className="card-header">
        <div className="card-title">
          <Gauge size={18} color="#38bdf8" />
          Environmental Sensor Readouts
        </div>
        <span className="badge badge-safe" style={{ textTransform: 'none' }}>
          Mode: {sensorData?.input_source === 'manual_simulation' ? 'Manual Simulation' : 'Hardware'}
        </span>
      </div>

      <div className="sensor-cards-grid">
        {/* Temperature */}
        <div className="sensor-card-mini" id="sensor-card-temp">
          <div className="label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Thermometer size={14} color="#f97316" /> Temperature
            </span>
            <span style={{ color: temp >= 70 ? '#ef4444' : temp >= 45 ? '#f59e0b' : '#34d399' }}>
              {temp >= 70 ? 'CRITICAL' : temp >= 45 ? 'ELEVATED' : 'NORMAL'}
            </span>
          </div>
          <div className="val">
            {temp.toFixed(1)} <span className="unit">°C</span>
          </div>
          <div className="progress-track" style={{ height: '6px' }}>
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, (temp / 120) * 100)}%`,
                backgroundColor: temp >= 70 ? '#ef4444' : temp >= 45 ? '#f97316' : '#10b981',
              }}
            />
          </div>
        </div>

        {/* Humidity */}
        <div className="sensor-card-mini" id="sensor-card-humidity">
          <div className="label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Droplets size={14} color="#38bdf8" /> Humidity
            </span>
            <span style={{ color: '#38bdf8' }}>DHT11</span>
          </div>
          <div className="val">
            {humidity.toFixed(1)} <span className="unit">%</span>
          </div>
          <div className="progress-track" style={{ height: '6px' }}>
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, humidity)}%`,
                backgroundColor: '#38bdf8',
              }}
            />
          </div>
        </div>

        {/* Smoke Level */}
        <div className="sensor-card-mini" id="sensor-card-smoke">
          <div className="label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <CloudFog size={14} color="#94a3b8" /> Smoke Level
            </span>
            <span style={{ color: smoke >= 70 ? '#ef4444' : smoke >= 40 ? '#f59e0b' : '#34d399' }}>
              {smoke >= 70 ? 'SEVERE' : smoke >= 40 ? 'ELEVATED' : 'CLEAR'}
            </span>
          </div>
          <div className="val">
            {smoke.toFixed(1)} <span className="unit">%</span>
          </div>
          <div className="progress-track" style={{ height: '6px' }}>
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, smoke)}%`,
                backgroundColor: smoke >= 70 ? '#ef4444' : smoke >= 40 ? '#f97316' : '#10b981',
              }}
            />
          </div>
        </div>

        {/* Gas Level */}
        <div className="sensor-card-mini" id="sensor-card-gas">
          <div className="label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Gauge size={14} color="#a855f7" /> Gas Level
            </span>
            <span style={{ color: gas >= 80 ? '#ef4444' : gas >= 45 ? '#f59e0b' : '#34d399' }}>
              {gas >= 80 ? 'HAZARDOUS' : gas >= 45 ? 'ELEVATED' : 'SAFE'}
            </span>
          </div>
          <div className="val">
            {gas.toFixed(1)} <span className="unit">%</span>
          </div>
          <div className="progress-track" style={{ height: '6px' }}>
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, gas)}%`,
                backgroundColor: gas >= 80 ? '#ef4444' : gas >= 45 ? '#f97316' : '#10b981',
              }}
            />
          </div>
        </div>

        {/* Flame Level */}
        <div className="sensor-card-mini" id="sensor-card-flame">
          <div className="label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Flame size={14} color="#ef4444" /> Flame Level
            </span>
            <span style={{ color: flame >= 60 ? '#ef4444' : flame >= 20 ? '#f59e0b' : '#34d399' }}>
              {flame >= 60 ? 'FLAME DETECTED' : flame >= 20 ? 'INTERMITTENT' : 'NONE'}
            </span>
          </div>
          <div className="val">
            {flame.toFixed(1)} <span className="unit">%</span>
          </div>
          <div className="progress-track" style={{ height: '6px' }}>
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, flame)}%`,
                backgroundColor: flame >= 60 ? '#dc2626' : flame >= 20 ? '#f97316' : '#10b981',
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
