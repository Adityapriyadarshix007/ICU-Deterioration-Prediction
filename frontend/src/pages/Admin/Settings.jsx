import React, { useState, useEffect } from 'react';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import api from '../../services/api';

function Settings() {
  const [settings, setSettings] = useState({});
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await api.getSystemSettings();
      setSettings(response.data);
    } catch (error) {
      toast.error('Failed to load settings');
      if (error.response?.status === 403) {
        navigate('/dashboard');
      }
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <AdminSidebar />
      <div className="ml-64 p-8">
        <h1 className="text-2xl font-bold mb-6">System Settings</h1>
        <div className="bg-white p-6 rounded-xl shadow max-w-2xl">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">App Name</label>
              <p className="text-lg font-semibold">{settings.app_name || 'ICU Predictor'}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Version</label>
              <p className="text-lg font-semibold">{settings.version || '1.0.0'}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Maintenance Mode</label>
              <p className={`text-lg font-semibold ${settings.maintenance_mode ? 'text-red-600' : 'text-green-600'}`}>
                {settings.maintenance_mode ? 'Enabled' : 'Disabled'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
