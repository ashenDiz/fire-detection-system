import React, { useState } from 'react';
import { Sliders, Send, AlertCircle, CheckCircle } from 'lucide-react';

export default function ManualSensorSimulator({ initialValues, onSubmit, isSubmitting }) {
  const [values, setValues] = useState({
    temperature: initialValues?.temperature ?? 28.0,
    humidity: initialValues?.humidity ?? 55.0,
    flame_level: initialValues?.flame_level ?? 0.0,
    smoke_level: initialValues?.smoke_level ?? 5.0,
    gas_level: initialValues?.gas_level ?? 4.0,
  });

  const [validationError, setValidationError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // Update when initialValues change (e.g. when scenario loaded)
  React.useEffect(() => {
    if (initialValues) {
      setValues({
        temperature: initialValues.temperature,
        humidity: initialValues.humidity,
        flame_level: initialValues.flame_level,
        smoke_level: initialValues.smoke_level,
        gas_level: initialValues.gas_level,
      });
      setSuccessMsg('');
      setValidationError('');
    }
  }, [initialValues]);

  const handleChange = (field, val) => {
    const num = parseFloat(val);
    setValues((prev) => ({
      ...prev,
      [field]: isNaN(num) ? 0 : num,
    }));
    setSuccessMsg('');
    setValidationError('');
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validation
    if (values.temperature < 0 || values.temperature > 120) {
      setValidationError('Temperature must be between 0 and 120 °C.');
      return;
    }
    if (values.humidity < 0 || values.humidity > 100) {
      setValidationError('Humidity must be between 0 and 100%.');
      return;
    }
    if (values.flame_level < 0 || values.flame_level > 100) {
      setValidationError('Flame level must be between 0 and 100%.');
      return;
    }
    if (values.smoke_level < 0 || values.smoke_level > 100) {
      setValidationError('Smoke level must be between 0 and 100%.');
      return;
    }
    if (values.gas_level < 0 || values.gas_level > 100) {
      setValidationError('Gas level must be between 0 and 100%.');
      return;
    }

    setValidationError('');
    onSubmit({
      ...values,
      input_source: 'manual_simulation',
      scenario_tag: 'MANUAL_INPUT',
    }).then(() => {
      setSuccessMsg('Sensor reading successfully persisted & risk re-assessed.');
      setTimeout(() => setSuccessMsg(''), 4000);
    });
  };

  return (
    <div className="card" id="manual-sensor-simulator-card">
      <div className="card-header">
        <div className="card-title">
          <Sliders size={18} color="#38bdf8" />
          Sensor Simulation / Manual Sensor Input
        </div>
        <span className="badge badge-safe" id="sensor-mode-indicator">
          Sensor Mode: Manual Simulation
        </span>
      </div>

      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
        Configure environmental telemetry parameters to simulate fire, combustion, or hazardous gas scenarios.
      </p>

      {validationError && (
        <div style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', color: '#fca5a5', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <AlertCircle size={15} />
          {validationError}
        </div>
      )}

      {successMsg && (
        <div style={{ background: 'rgba(16,185,129,0.15)', border: '1px solid #10b981', color: '#6ee7b7', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <CheckCircle size={15} />
          {successMsg}
        </div>
      )}

      <form onSubmit={handleSubmit} className="simulator-form">
        {/* Temperature */}
        <div className="slider-group">
          <div className="slider-header">
            <label htmlFor="input-temp">Temperature (°C) [0 - 120 °C]</label>
            <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{values.temperature} °C</span>
          </div>
          <div className="slider-controls">
            <input
              id="slider-temp"
              type="range"
              min="0"
              max="120"
              step="0.5"
              value={values.temperature}
              onChange={(e) => handleChange('temperature', e.target.value)}
            />
            <input
              id="input-temp"
              type="number"
              min="0"
              max="120"
              step="0.1"
              value={values.temperature}
              onChange={(e) => handleChange('temperature', e.target.value)}
            />
          </div>
        </div>

        {/* Humidity */}
        <div className="slider-group">
          <div className="slider-header">
            <label htmlFor="input-humidity">Humidity (%) [0 - 100 %]</label>
            <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{values.humidity} %</span>
          </div>
          <div className="slider-controls">
            <input
              id="slider-humidity"
              type="range"
              min="0"
              max="100"
              step="1"
              value={values.humidity}
              onChange={(e) => handleChange('humidity', e.target.value)}
            />
            <input
              id="input-humidity"
              type="number"
              min="0"
              max="100"
              step="0.5"
              value={values.humidity}
              onChange={(e) => handleChange('humidity', e.target.value)}
            />
          </div>
        </div>

        {/* Smoke Level */}
        <div className="slider-group">
          <div className="slider-header">
            <label htmlFor="input-smoke">Smoke Level (%) [0 - 100 %]</label>
            <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{values.smoke_level} %</span>
          </div>
          <div className="slider-controls">
            <input
              id="slider-smoke"
              type="range"
              min="0"
              max="100"
              step="1"
              value={values.smoke_level}
              onChange={(e) => handleChange('smoke_level', e.target.value)}
            />
            <input
              id="input-smoke"
              type="number"
              min="0"
              max="100"
              step="0.5"
              value={values.smoke_level}
              onChange={(e) => handleChange('smoke_level', e.target.value)}
            />
          </div>
        </div>

        {/* Gas Level */}
        <div className="slider-group">
          <div className="slider-header">
            <label htmlFor="input-gas">Gas Level (%) [0 - 100 %]</label>
            <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{values.gas_level} %</span>
          </div>
          <div className="slider-controls">
            <input
              id="slider-gas"
              type="range"
              min="0"
              max="100"
              step="1"
              value={values.gas_level}
              onChange={(e) => handleChange('gas_level', e.target.value)}
            />
            <input
              id="input-gas"
              type="number"
              min="0"
              max="100"
              step="0.5"
              value={values.gas_level}
              onChange={(e) => handleChange('gas_level', e.target.value)}
            />
          </div>
        </div>

        {/* Flame Level */}
        <div className="slider-group">
          <div className="slider-header">
            <label htmlFor="input-flame">Flame Level (%) [0 - 100 %]</label>
            <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{values.flame_level} %</span>
          </div>
          <div className="slider-controls">
            <input
              id="slider-flame"
              type="range"
              min="0"
              max="100"
              step="1"
              value={values.flame_level}
              onChange={(e) => handleChange('flame_level', e.target.value)}
            />
            <input
              id="input-flame"
              type="number"
              min="0"
              max="100"
              step="0.5"
              value={values.flame_level}
              onChange={(e) => handleChange('flame_level', e.target.value)}
            />
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          id="btn-submit-sensor"
          disabled={isSubmitting}
          style={{ width: '100%', marginTop: '0.5rem' }}
        >
          <Send size={16} />
          {isSubmitting ? 'Evaluating & Saving...' : 'Submit Sensor Reading'}
        </button>
      </form>
    </div>
  );
}
