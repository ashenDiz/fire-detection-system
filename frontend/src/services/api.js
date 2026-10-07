import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 5000,
});

export const systemApi = {
  getStatus: () => apiClient.get('/system/status').then((res) => res.data),
};

export const sensorsApi = {
  submitReading: (data) => apiClient.post('/sensors/readings', data).then((res) => res.data),
  getLatest: () => apiClient.get('/sensors/latest').then((res) => res.data),
  getHistory: (limit = 10) => apiClient.get(`/sensors/history?limit=${limit}`).then((res) => res.data),
};

export const riskApi = {
  getCurrent: () => apiClient.get('/risk/current').then((res) => res.data),
};

export const scenariosApi = {
  listScenarios: () => apiClient.get('/scenarios').then((res) => res.data),
};

export const alertsApi = {
  getAlerts: (unacknowledgedOnly = false) =>
    apiClient.get(`/alerts?unacknowledged_only=${unacknowledgedOnly}`).then((res) => res.data),
  acknowledge: (alertId) =>
    apiClient.post(`/alerts/${alertId}/acknowledge`).then((res) => res.data),
};

export const cameraApi = {
  getStatus: () => apiClient.get('/camera/status').then((res) => res.data),
  start: (cameraIndex = null) =>
    apiClient
      .post(
        '/camera/start',
        cameraIndex !== null ? { camera_index: cameraIndex } : {},
        { timeout: 10000 }
      )
      .then((res) => res.data),
  stop: () => apiClient.post('/camera/stop', {}, { timeout: 10000 }).then((res) => res.data),
  getStreamUrl: () => `${API_BASE_URL}/camera/stream`,
};

export const detectionApi = {
  getPersonStatus: () => apiClient.get('/detection/person/status').then((res) => res.data),
  getPersonLatest: () => apiClient.get('/detection/person/latest').then((res) => res.data),
};

export const emergencyApi = {
  getStatus: () => apiClient.get('/emergency/status').then((res) => res.data),
  getEmailStatus: () => apiClient.get('/emergency/email/status').then((res) => res.data),
};

