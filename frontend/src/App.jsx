import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { LoadingProvider } from './context/LoadingContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import ChangePassword from './pages/ChangePassword';
import Dashboard from './pages/Dashboard';
import PatientSearch from './pages/PatientSearch';
import PatientDetail from './pages/PatientDetail';
import AdminDashboard from './pages/Admin/AdminDashboard';
import Patients from './pages/Admin/Patients';
import Users from './pages/Admin/Users';
import Analytics from './pages/Admin/Analytics';
import Settings from './pages/Admin/Settings';
import Logs from './pages/Admin/Logs';
import './index.css';

// Get Google Client ID from environment variable
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

// Check if client ID is configured
if (!GOOGLE_CLIENT_ID || GOOGLE_CLIENT_ID === 'your-google-client-id.apps.googleusercontent.com') {
  console.warn('⚠️ Google Client ID not configured. Google login will not work.');
  console.warn('   Add VITE_GOOGLE_CLIENT_ID to your .env file');
}

function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID || ''}>
      <ThemeProvider>
        <LoadingProvider>
          <AuthProvider>
            <BrowserRouter>
              <Toaster 
                position="top-right"
                toastOptions={{
                  duration: 2000, // Changed from 4000 to 2000 (2 seconds)
                  style: {
                    background: '#363636',
                    color: '#fff',
                    borderRadius: '8px',
                    padding: '12px 16px',
                  },
                  success: {
                    duration: 2000,
                    iconTheme: {
                      primary: '#22c55e',
                      secondary: '#fff',
                    },
                  },
                  error: {
                    duration: 2000,
                    iconTheme: {
                      primary: '#ef4444',
                      secondary: '#fff',
                    },
                  },
                }}
              />
              <Routes>
                {/* Public Routes */}
                <Route path="/login" element={<Login />} />
                <Route path="/change-password" element={<ChangePassword />} />
                <Route path="/" element={<Navigate to="/dashboard" />} />
                
                {/* Protected Routes */}
                <Route element={<ProtectedRoute />}>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/patients" element={<PatientSearch />} />
                  <Route path="/patients/:id" element={<PatientDetail />} />
                  <Route path="/admin" element={<AdminDashboard />} />
                  <Route path="/admin/patients" element={<Patients />} />
                  <Route path="/admin/users" element={<Users />} />
                  <Route path="/admin/analytics" element={<Analytics />} />
                  <Route path="/admin/settings" element={<Settings />} />
                  <Route path="/admin/logs" element={<Logs />} />
                </Route>
              </Routes>
            </BrowserRouter>
          </AuthProvider>
        </LoadingProvider>
      </ThemeProvider>
    </GoogleOAuthProvider>
  );
}

export default App;
