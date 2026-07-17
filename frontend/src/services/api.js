import axios from 'axios';
import toast from 'react-hot-toast';

const API_URL = '/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
      toast.error('Session expired. Please login again.');
    }
    return Promise.reject(error);
  }
);

// ============================================================
// Auth APIs
// ============================================================

const auth = {
  login: (username, password) => api.post('/auth/login/json', { username, password }),
  register: (data) => api.post('/auth/register', data),
  googleLogin: (idToken) => api.post('/auth/google/login', { id_token: idToken }),
  getCurrentUser: () => api.get('/auth/me'),
  logout: () => api.post('/auth/logout'),
};

// ============================================================
// Prediction APIs
// ============================================================

const predictions = {
  predict: (data) => api.post('/predict/public', data),
  getPatientPredictions: (patientId) => api.get(`/predict/patient/${patientId}`),
  getRecentPredictions: () => api.get('/predict/recent'),
};

// ============================================================
// Dashboard APIs
// ============================================================

const dashboard = {
  getDashboardStats: () => api.get('/dashboard/stats'),
};

// ============================================================
// Admin APIs
// ============================================================

const admin = {
  // Users
  getUsers: () => api.get('/admin/users'),
  deleteUser: (userId) => api.delete(`/admin/users/${userId}`),
  makeAdmin: (userId) => api.post(`/admin/make-admin/${userId}`),
  
  // Patients
  getPatients: (search = '') => {
    const url = search ? `/admin/patients?search=${encodeURIComponent(search)}` : '/admin/patients';
    return api.get(url);
  },
  createPatient: (data) => api.post('/admin/patients', data),
  updatePatient: (patientId, data) => api.put(`/admin/patients/${patientId}`, data),
  deletePatient: (patientId) => api.delete(`/admin/patients/${patientId}`),
  importPatients: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/admin/patients/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  exportPatients: () => api.get('/admin/patients/export', { responseType: 'blob' }),
  getPatientStats: () => api.get('/admin/patients/stats'),
  
  // Analytics
  getOverview: () => api.get('/admin/overview'),
  getActivityLogs: () => api.get('/admin/logs'),
  
  // Settings
  getSettings: () => api.get('/admin/settings'),
  updateSettings: (settings) => api.put('/admin/settings', settings),
};

// ============================================================
// Export All APIs
// ============================================================

export default {
  ...auth,
  ...predictions,
  ...dashboard,
  ...admin,
};
