import React from 'react';
import { AlertCircle, ShieldAlert } from 'lucide-react';

export default function DisclaimerFooter() {
  return (
    <footer className="system-footer" id="system-disclaimer-footer">
      <div className="disclaimer-box">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', color: '#f59e0b', fontWeight: 600, marginBottom: '0.2rem' }}>
          <ShieldAlert size={15} />
          Academic Prototype Safety & Regulatory Disclaimer
        </div>
        <p>
          This system is an academic research prototype developed for multi-modal sensor fusion and computer vision investigation.
          It is <strong>not a replacement</strong> for a certified fire detection, fire alarm, evacuation, or life-safety system.
        </p>
      </div>

      <div style={{ color: 'var(--text-muted)' }}>
        Environmental sensor values are currently provided through manual simulation for prototype testing.
        The backend architecture features an abstract <code>SensorDataProvider</code> interface designed for plug-and-play integration with ESP32 / Arduino hardware telemetry.
      </div>
    </footer>
  );
}
