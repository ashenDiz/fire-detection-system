import React from 'react';
import { Bell, Check, AlertOctagon, CheckCircle2 } from 'lucide-react';

export default function ActiveAlerts({ alerts, onAcknowledge }) {
  const unacked = alerts?.filter((a) => !a.acknowledged) || [];

  return (
    <div className="card" id="alerts-card">
      <div className="card-header">
        <div className="card-title">
          <Bell size={18} color="#ef4444" />
          Active Incident Alerts & Notifications
        </div>
        <span className="badge" style={{ background: unacked.length > 0 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.1)', color: unacked.length > 0 ? '#ef4444' : '#10b981', border: unacked.length > 0 ? '1px solid #ef4444' : '1px solid #10b981' }}>
          {unacked.length} Unacknowledged
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }} id="alerts-list">
        {alerts && alerts.length > 0 ? (
          alerts.slice(0, 5).map((alert) => (
            <div
              key={alert.id}
              className="alert-banner"
              style={{
                borderColor: alert.acknowledged ? '#334155' : alert.alert_type === 'CRITICAL' ? '#dc2626' : '#f97316',
                background: alert.acknowledged ? 'rgba(15, 23, 42, 0.4)' : 'rgba(220, 38, 38, 0.08)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem' }}>
                <AlertOctagon
                  size={18}
                  color={alert.acknowledged ? '#64748b' : alert.alert_type === 'CRITICAL' ? '#ef4444' : '#f97316'}
                  style={{ flexShrink: 0, marginTop: '2px' }}
                />
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                    <strong style={{ fontSize: '0.8rem', color: alert.acknowledged ? '#94a3b8' : '#ffffff' }}>
                      [{alert.alert_type}] Risk Score: {alert.risk_score}
                    </strong>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {new Date(alert.created_at).toLocaleTimeString()}
                    </span>
                    {alert.acknowledged && (
                      <span className="badge badge-safe" style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                        Acknowledged
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.775rem', color: alert.acknowledged ? '#94a3b8' : '#cbd5e1' }}>
                    {alert.message}
                  </div>
                </div>
              </div>

              {!alert.acknowledged && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => onAcknowledge(alert.id)}
                  style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', flexShrink: 0 }}
                >
                  <Check size={14} />
                  Acknowledge
                </button>
              )}
            </div>
          ))
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic', padding: '0.5rem' }}>
            No incident alerts logged. Monitored environment operating safely.
          </div>
        )}
      </div>
    </div>
  );
}
