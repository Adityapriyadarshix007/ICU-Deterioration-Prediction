import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider } from './context/AuthContext.jsx';
import ProtectedRoute from './components/ProtectedRoute.jsx';
import Login from './pages/Login.jsx';
import Dashboard from './pages/Dashboard.jsx';
import PatientSearch from './pages/PatientSearch.jsx';
import PatientDetail from './pages/PatientDetail.jsx';
// Admin Pages
import AdminDashboard from './pages/Admin/AdminDashboard.jsx';
import Patients from './pages/Admin/Patients.jsx';
import Users from './pages/Admin/Users.jsx';
import Analytics from './pages/Admin/Analytics.jsx';
import Settings from './pages/Admin/Settings.jsx';
import Logs from './pages/Admin/Logs.jsx';

const GOOGLE_CLIENT_ID = '470384815602-voh3j0i6bsdnupjb4s2hvlhmi3172o0g.apps.googleusercontent.com';

function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <AuthProvider>
        <BrowserRouter>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={<Navigate to="/dashboard" />} />
            <Route element={<ProtectedRoute />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/patients" element={<PatientSearch />} />
              <Route path="/patients/:id" element={<PatientDetail />} />
              {/* Admin Routes */}
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
    </GoogleOAuthProvider>
  );
}

export default App;
