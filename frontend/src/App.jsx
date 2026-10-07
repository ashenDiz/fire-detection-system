import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import SystemStatusPanel from './components/SystemStatusPanel';
import RiskOverview from './components/RiskOverview';
import SensorCards from './components/SensorCards';
import ManualSensorSimulator from './components/ManualSensorSimulator';
import ResearchScenarios from './components/ResearchScenarios';
import ContributingFactors from './components/ContributingFactors';
import CameraFeed from './components/CameraFeed';
import ActiveAlerts from './components/ActiveAlerts';
import EmergencyAlertBanner from './components/EmergencyAlertBanner';

import DisclaimerFooter from './components/DisclaimerFooter';
import { systemApi, sensorsApi, riskApi, scenariosApi, alertsApi } from './services/api';

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [sensorData, setSensorData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isBackendOnline, setIsBackendOnline] = useState(true);

  // Initial and Polling Data Fetch
  const refreshData = useCallback(async () => {
    try {
      const [statusRes, latestSensorRes, riskRes, scenariosRes, alertsRes] = await Promise.all([
        systemApi.getStatus().catch(() => null),
        sensorsApi.getLatest().catch(() => null),
        riskApi.getCurrent().catch(() => null),
        scenariosApi.listScenarios().catch(() => []),
        alertsApi.getAlerts().catch(() => []),
      ]);

      if (statusRes) {
        setSystemStatus(statusRes);
        setIsBackendOnline(true);
      } else {
        setIsBackendOnline(false);
      }

      if (latestSensorRes) setSensorData(latestSensorRes);
      if (riskRes) setRiskData(riskRes);
      if (scenariosRes && scenariosRes.length > 0) setScenarios(scenariosRes);
      if (alertsRes) setAlerts(alertsRes);
    } catch (err) {
      console.error('Error polling dashboard telemetry:', err);
      setIsBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 4000);
    return () => clearInterval(interval);
  }, [refreshData]);

  // Handle camera status transition notifications from CameraFeed
  const handleCameraStatusChange = useCallback((cameraData) => {
    if (cameraData) {
      setSystemStatus((prev) => (prev ? {
        ...prev,
        camera: cameraData.connected ? 'Connected' : (cameraData.status?.startsWith('Error') ? cameraData.status : 'Disconnected'),
      } : prev));
    }
    refreshData();
  }, [refreshData]);

  // Handle sensor reading submission
  const handleSensorSubmit = async (data) => {
    setIsSubmitting(true);
    try {
      const result = await sensorsApi.submitReading(data);
      if (result.reading) setSensorData(result.reading);
      if (result.risk) setRiskData(result.risk);
      if (result.active_alert) {
        setAlerts((prev) => [result.active_alert, ...prev.filter((a) => a.id !== result.active_alert.id)]);
      }
      setIsBackendOnline(true);
    } catch (err) {
      console.error('Submission failed:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle loading scenario preset
  const handleLoadScenario = (preset) => {
    const newValues = {
      temperature: preset.temperature,
      humidity: preset.humidity,
      smoke_level: preset.smoke_level,
      gas_level: preset.gas_level,
      flame_level: preset.flame_level,
      input_source: 'manual_simulation',
      scenario_tag: preset.id,
    };
    handleSensorSubmit(newValues);
  };

  // Handle acknowledging an alert
  const handleAcknowledgeAlert = async (alertId) => {
    try {
      const updated = await alertsApi.acknowledge(alertId);
      setAlerts((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  return (
    <div className="app-container">
      <Header systemOnline={isBackendOnline} riskLevel={riskData?.risk_level} />
      <EmergencyAlertBanner />
      <SystemStatusPanel status={systemStatus} />

      <main className="dashboard-grid">
        {/* Left Column: Assessment & Telemetry Metrics */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <RiskOverview risk={riskData} />
          <ContributingFactors
            factors={riskData?.contributing_factors}
            riskLevel={riskData?.risk_level}
          />
          <SensorCards sensorData={sensorData} />
        </section>

        {/* Right Column: Simulation Controls, Scenarios, and Camera Status */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <ManualSensorSimulator
            initialValues={sensorData}
            onSubmit={handleSensorSubmit}
            isSubmitting={isSubmitting}
          />
          <ResearchScenarios
            scenarios={scenarios}
            onLoadScenario={handleLoadScenario}
          />
          <CameraFeed onStatusChange={handleCameraStatusChange} />
        </section>

      </main>

      <section>
        <ActiveAlerts alerts={alerts} onAcknowledge={handleAcknowledgeAlert} />
      </section>

      <DisclaimerFooter />
    </div>
  );
}
