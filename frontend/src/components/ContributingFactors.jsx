import React from 'react';
import { HelpCircle, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ContributingFactors({ factors, riskLevel }) {
  const isSafe = riskLevel === 'SAFE';

  return (
    <div className="card" id="contributing-factors-card">
      <div className="card-header">
        <div className="card-title">
          <HelpCircle size={18} color="#38bdf8" />
          Explainable Multi-Source Fusion Analysis
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Why is this classified as <strong>{riskLevel}</strong>?
        </span>
      </div>

      <div className="factors-list" id="factors-container">
        {factors && factors.length > 0 ? (
          factors.map((factor, idx) => (
            <div
              key={idx}
              className="factor-item"
              style={{
                borderLeftColor: isSafe ? '#10b981' : riskLevel === 'CRITICAL' ? '#dc2626' : '#f97316',
              }}
            >
              {isSafe ? (
                <CheckCircle2 size={16} color="#10b981" style={{ flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <AlertCircle size={16} color={riskLevel === 'CRITICAL' ? '#ef4444' : '#f97316'} style={{ flexShrink: 0, marginTop: '2px' }} />
              )}
              <span>{factor}</span>
            </div>
          ))
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic' }}>
            No anomaly factors identified. All sensor modalities operating at baseline limits.
          </div>
        )}
      </div>
    </div>
  );
}
