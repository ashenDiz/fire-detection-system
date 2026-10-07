import React, { useState } from 'react';
import { Beaker, ArrowRightCircle } from 'lucide-react';

export default function ResearchScenarios({ scenarios, onLoadScenario }) {
  const [selectedScenarioId, setSelectedScenarioId] = useState(scenarios?.[0]?.id || 'NORMAL');

  const activePreset = scenarios?.find((s) => s.id === selectedScenarioId) || scenarios?.[0];

  const handleSelect = (id) => {
    setSelectedScenarioId(id);
  };

  const handleApply = () => {
    if (activePreset) {
      onLoadScenario(activePreset);
    }
  };

  return (
    <div className="card" id="research-scenarios-panel">
      <div className="card-header">
        <div className="card-title">
          <Beaker size={18} color="#a855f7" />
          Research Test Scenarios
        </div>
        <span className="badge badge-caution">Repeatable Benchmark</span>
      </div>

      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
        Select a predefined research test condition and explicitly click <strong>Load Scenario</strong> to populate the simulator.
      </p>

      <div className="scenarios-grid">
        {scenarios?.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`btn-scenario ${selectedScenarioId === s.id ? 'active' : ''}`}
            id={`btn-scenario-${s.id.toLowerCase()}`}
            onClick={() => handleSelect(s.id)}
          >
            <strong style={{ fontSize: '0.8rem' }}>{s.name}</strong>
            <span style={{ fontSize: '0.675rem', color: 'var(--text-muted)' }}>
              T:{s.temperature}°C • S:{s.smoke_level}% • G:{s.gas_level}% • F:{s.flame_level}%
            </span>
          </button>
        ))}
      </div>

      {activePreset && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.7)',
            padding: '0.75rem',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.4rem',
          }}
        >
          <div style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 600 }}>
            {activePreset.name} Target Telemetry:
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            {activePreset.description}
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.725rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', marginTop: '0.2rem' }}>
            <span>Temp: {activePreset.temperature}°C</span>
            <span>Humidity: {activePreset.humidity}%</span>
            <span>Smoke: {activePreset.smoke_level}%</span>
            <span>Gas: {activePreset.gas_level}%</span>
            <span>Flame: {activePreset.flame_level}%</span>
          </div>
        </div>
      )}

      <button
        type="button"
        className="btn btn-secondary"
        id="btn-load-scenario"
        onClick={handleApply}
        style={{ width: '100%', borderColor: '#a855f7', color: '#c084fc' }}
      >
        <ArrowRightCircle size={16} />
        Load Scenario into Simulator
      </button>
    </div>
  );
}
