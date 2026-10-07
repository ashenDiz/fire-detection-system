import React from 'react';
import { Shield, AlertTriangle, Users, Flame, Info } from 'lucide-react';

export default function RiskOverview({ risk }) {
  const score = risk?.overall_risk_score ?? 0;
  const level = risk?.risk_level || 'SAFE';
  const priority = risk?.emergency_priority || 'LOW';

  const getLevelColor = (lvl) => {
    switch (lvl) {
      case 'CRITICAL':
        return '#dc2626';
      case 'HIGH RISK':
        return '#ef4444';
      case 'WARNING':
        return '#f97316';
      case 'CAUTION':
        return '#f59e0b';
      case 'SAFE':
      default:
        return '#10b981';
    }
  };

  const getBadgeClass = (lvl) => {
    switch (lvl) {
      case 'CRITICAL':
        return 'badge-critical';
      case 'HIGH RISK':
        return 'badge-high';
      case 'WARNING':
        return 'badge-warning';
      case 'CAUTION':
        return 'badge-caution';
      case 'SAFE':
      default:
        return 'badge-safe';
    }
  };

  const levelColor = getLevelColor(level);

  return (
    <div className="card" id="risk-overview-card">
      <div className="card-header">
        <div className="card-title">
          <Shield size={18} color={levelColor} />
          Multi-Source Fire Risk Assessment
        </div>
        <div className="flex items-center gap-2">
          <span className={`badge ${getBadgeClass(level)}`} id="current-risk-badge">
            {level}
          </span>
        </div>
      </div>

      <div className="risk-meter-container">
        <div className="risk-score-display">
          <div>
            <span className="score-number" style={{ color: levelColor }} id="risk-score-value">
              {score.toFixed(1)}
            </span>
            <span className="score-unit">/ 100</span>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span className="status-label">Emergency Priority</span>
            <div style={{ marginTop: '0.25rem' }}>
              <span className={`badge ${getBadgeClass(priority === 'CRITICAL' ? 'CRITICAL' : priority === 'HIGH' ? 'HIGH RISK' : 'SAFE')}`} id="emergency-priority-badge">
                {priority} PRIORITY
              </span>
            </div>
          </div>
        </div>

        <div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(100, Math.max(0, score))}%`,
                backgroundColor: levelColor,
              }}
            />
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.675rem',
              color: 'var(--text-muted)',
              marginTop: '0.35rem',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <span>SAFE (0-29)</span>
            <span>CAUTION (30-49)</span>
            <span>WARNING (50-69)</span>
            <span>HIGH RISK (70-84)</span>
            <span>CRITICAL (85-100)</span>
          </div>
        </div>

        {/* Sub-Risk Normalization Breakdown */}
        <div className="sub-risk-grid">
          <div className="sub-risk-item">
            <div className="sub-risk-label">Thermal Risk</div>
            <div className="sub-risk-val" style={{ color: getLevelColor(risk?.temp_risk > 50 ? 'WARNING' : 'SAFE') }}>
              {risk?.temp_risk?.toFixed(1) ?? '0.0'}%
            </div>
          </div>

          <div className="sub-risk-item">
            <div className="sub-risk-label">Smoke Risk</div>
            <div className="sub-risk-val" style={{ color: getLevelColor(risk?.smoke_risk > 50 ? 'WARNING' : 'SAFE') }}>
              {risk?.smoke_risk?.toFixed(1) ?? '0.0'}%
            </div>
          </div>

          <div className="sub-risk-item">
            <div className="sub-risk-label">Gas Risk</div>
            <div className="sub-risk-val" style={{ color: getLevelColor(risk?.gas_risk > 50 ? 'WARNING' : 'SAFE') }}>
              {risk?.gas_risk?.toFixed(1) ?? '0.0'}%
            </div>
          </div>

          <div className="sub-risk-item">
            <div className="sub-risk-label">Flame Risk</div>
            <div className="sub-risk-val" style={{ color: getLevelColor(risk?.flame_risk > 50 ? 'WARNING' : 'SAFE') }}>
              {risk?.flame_risk?.toFixed(1) ?? '0.0'}%
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Users size={15} color="#38bdf8" />
            <span>People currently detected by camera:</span>
            <strong style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }} id="people-count-value">
              {risk?.people_detection_available && risk?.people_detected !== null && risk?.people_detected !== undefined
                ? risk.people_detected
                : 'Unavailable'}
            </strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Visual AI Weight: </span>
            <span style={{ color: '#94a3b8' }}>Phase 1 Standby</span>
          </div>
        </div>
      </div>
    </div>
  );
}
