import React, { useState, useEffect } from 'react';
import { Flame, ShieldAlert, Cpu } from 'lucide-react';

export default function Header({ systemOnline, riskLevel }) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="system-header">
      <div className="header-title-block">
        <h1>
          <Flame size={26} color="#f97316" />
          Intelligent Multi-Modal Fire Detection & Emergency Alert System
        </h1>
        <div className="subtitle">
          Sensor Fusion & Computer Vision Prototype • Academic Research Platform
        </div>
      </div>
      <div className="header-meta">
        <div className="clock-display" id="system-clock">
          {timeStr || 'Syncing clock...'}
        </div>
        <div
          className={`badge ${
            systemOnline ? 'badge-safe' : 'badge-critical'
          }`}
          id="system-backend-badge"
        >
          <Cpu size={14} />
          {systemOnline ? 'System Online' : 'Backend Offline'}
        </div>
      </div>
    </header>
  );
}
