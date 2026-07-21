import axios from 'axios';
import toast from 'react-hot-toast';

// Use environment variable for API URL with fallback for development
const API_URL = import.meta.env.VITE_API_URL || '/api';

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

// Response interceptor - Don't intercept login 401s
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Check if it's a login request
    const isAuthLoginRequest = 
      error.config?.url?.includes('/auth/login') ||
      error.config?.url?.includes('/auth/google/login');

    // Only handle 401 for non-login requests and if user has a token
    if (
      error.response?.status === 401 &&
      !isAuthLoginRequest &&
      localStorage.getItem('access_token')
    ) {
      localStorage.removeItem('access_token');
      toast.error('Session expired. Please login again.');
      window.location.href = '/login';
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
  changePassword: (currentPassword, newPassword) => 
    api.post('/auth/change-password', { 
      current_password: currentPassword, 
      new_password: newPassword 
    }),
  resetPasswordByEmail: (email, newPassword) => 
    api.post('/auth/reset-password-email', { 
      email: email,
      new_password: newPassword 
    }),
};

// ============================================================
// Patient APIs (Public - accessible by any authenticated user)
// ============================================================

const patients = {
  // Public patient endpoints (for doctors)
  getPatients: async (search = '', limit = 100, skip = 0) => {
    try {
      const url = search 
        ? `/patients?search=${encodeURIComponent(search)}&limit=${limit}&skip=${skip}`
        : `/patients?limit=${limit}&skip=${skip}`;
      const response = await api.get(url);
      return response;
    } catch (error) {
      console.error('Failed to load patients:', error);
      throw error;
    }
  },
  getPatient: async (patientId) => {
    try {
      const response = await api.get(`/patients/${patientId}`);
      return response;
    } catch (error) {
      console.error('Failed to load patient:', error);
      throw error;
    }
  },
};

// ============================================================
// Admin Patient APIs (Admin only - full CRUD)
// ============================================================

const adminPatients = {
  getAdminPatients: async (search = '', limit = 100, skip = 0) => {
    try {
      const url = search 
        ? `/admin/patients?search=${encodeURIComponent(search)}&limit=${limit}&skip=${skip}`
        : `/admin/patients?limit=${limit}&skip=${skip}`;
      const response = await api.get(url);
      return response;
    } catch (error) {
      console.error('Failed to load admin patients:', error);
      throw error;
    }
  },
  createAdminPatient: async (data) => {
    try {
      // Ensure patient_id is present
      if (!data.patient_id) {
        data.patient_id = `PAT-${Date.now().toString().slice(-8)}`;
      }
      const response = await api.post('/admin/patients', data);
      return response;
    } catch (error) {
      console.error('Failed to create patient:', error);
      throw error;
    }
  },
  updateAdminPatient: async (patientId, data) => {
    try {
      const response = await api.put(`/admin/patients/${patientId}`, data);
      return response;
    } catch (error) {
      console.error('Failed to update patient:', error);
      throw error;
    }
  },
  deleteAdminPatient: async (patientId) => {
    try {
      const response = await api.delete(`/admin/patients/${patientId}`);
      return response;
    } catch (error) {
      console.error('Failed to delete patient:', error);
      throw error;
    }
  },
  importPatients: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await api.post('/admin/patients/import', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      return response;
    } catch (error) {
      console.error('Failed to import patients:', error);
      throw error;
    }
  },
  exportPatients: async () => {
    try {
      const response = await api.get('/admin/patients/export', { responseType: 'blob' });
      return response;
    } catch (error) {
      console.error('Failed to export patients:', error);
      throw error;
    }
  },
  getPatientStats: async () => {
    try {
      const response = await api.get('/admin/patients/stats');
      return response;
    } catch (error) {
      console.error('Failed to get patient stats:', error);
      throw error;
    }
  },
};

// ============================================================
// Prediction APIs
// ============================================================

const predictions = {
  predict: async (data) => {
    try {
      const response = await api.post('/predict/public', data);
      return response;
    } catch (error) {
      console.error('Prediction API error:', error);
      throw error;
    }
  },
  getPatientPredictions: async (patientId) => {
    try {
      const response = await api.get(`/predict/patient/${patientId}`);
      return response;
    } catch (error) {
      console.error('Failed to load patient predictions:', error);
      return { data: [] };
    }
  },
  getRecentPredictions: async () => {
    try {
      const response = await api.get('/predict/recent');
      return response;
    } catch (error) {
      console.error('Failed to load recent predictions:', error);
      return { data: [] };
    }
  },
};

// ============================================================
// Dashboard APIs
// ============================================================

const dashboard = {
  getDashboardStats: async () => {
    try {
      const response = await api.get('/dashboard/stats');
      return response;
    } catch (error) {
      console.error('Failed to load dashboard stats:', error);
      return { 
        data: {
          total_predictions: 0,
          high_risk_patients: 0,
          critical_alerts: 0,
          avg_risk_score: 0,
          recent_predictions: []
        }
      };
    }
  },
};

// ============================================================
// Settings APIs (Admin only)
// ============================================================

const settings = {
  getSettings: async () => {
    try {
      const response = await api.get('/admin/settings');
      return response;
    } catch (error) {
      console.error('Failed to load settings:', error);
      throw error;
    }
  },
  updateSettings: async (settingsData) => {
    try {
      const response = await api.put('/admin/settings', settingsData);
      return response;
    } catch (error) {
      console.error('Failed to update settings:', error);
      throw error;
    }
  },
};

// ============================================================
// Admin User APIs (Admin only)
// ============================================================

const adminUsers = {
  getUsers: async () => {
    try {
      const response = await api.get('/admin/users');
      return response;
    } catch (error) {
      console.error('Failed to load users:', error);
      return { data: { users: [] } };
    }
  },
  deleteUser: (userId) => api.delete(`/admin/users/${userId}`),
  makeAdmin: (userId) => api.post(`/admin/make-admin/${userId}`),
};

// ============================================================
// Admin Analytics APIs (Admin only)
// ============================================================

const adminAnalytics = {
  getOverview: async () => {
    try {
      const response = await api.get('/admin/overview');
      return response;
    } catch (error) {
      console.error('Failed to load overview:', error);
      return { 
        data: {
          users: { total: 0, active: 0 },
          predictions: { total: 0, weekly: 0, high_risk: 0, critical: 0, avg_risk: 0 }
        }
      };
    }
  },
  getActivityLogs: async () => {
    try {
      const response = await api.get('/admin/logs');
      return response;
    } catch (error) {
      console.error('Failed to load logs:', error);
      return { data: { logs: [] } };
    }
  },
};

// ============================================================
// Export All APIs (Backward compatible)
// ============================================================

export default {
  ...auth,
  ...patients,
  ...predictions,
  ...dashboard,
  ...settings,
  // Admin endpoints (for backward compatibility)
  getAdminPatients: adminPatients.getAdminPatients,
  createPatient: adminPatients.createAdminPatient,
  updatePatient: adminPatients.updateAdminPatient,
  deletePatient: adminPatients.deleteAdminPatient,
  importPatients: adminPatients.importPatients,
  exportPatients: adminPatients.exportPatients,
  getPatientStats: adminPatients.getPatientStats,
  getUsers: adminUsers.getUsers,
  deleteUser: adminUsers.deleteUser,
  makeAdmin: adminUsers.makeAdmin,
  getOverview: adminAnalytics.getOverview,
  getActivityLogs: adminAnalytics.getActivityLogs,
  // Direct admin patient exports
  adminPatients,
  adminUsers,
  adminAnalytics,
};