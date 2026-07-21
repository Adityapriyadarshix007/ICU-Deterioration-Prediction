import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';
import { Save, Settings as SettingsIcon, Globe, Shield, Bell, Database, RefreshCw, Loader2 } from 'lucide-react';

function Settings() {
  const [settings, setSettings] = useState({
    app_name: 'ICU Predictor',
    version: '1.0.0',
    maintenance_mode: false,
    allow_registration: true,
    max_predictions_per_day: 100,
    model_threshold: 0.459,
    prediction_window: 24,
    alert_frequency: 60,
    auto_prediction: true,
    imputation_method: 'knn',
    email_notifications: true,
    data_retention_days: 30
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const response = await api.getSettings();
      console.log('Settings response:', response.data);
      
      // Merge response with default settings
      if (response.data) {
        setSettings(prev => ({
          ...prev,
          ...response.data
        }));
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
      if (error.response?.status === 403) {
        toast.error('Admin access required');
        navigate('/dashboard');
      } else if (error.response?.status === 401) {
        toast.error('Please login again');
        navigate('/login');
      } else {
        toast.error('Unable to load settings. Using default values.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    
    try {
      await api.updateSettings(settings);
      toast.success('Settings saved successfully!');
    } catch (error) {
      console.error('Failed to save settings:', error);
      if (error.response?.status === 403) {
        toast.error('Admin access required');
      } else {
        toast.error('Failed to save settings');
      }
    } finally {
      setSaving(false);
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
        <Breadcrumb />
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold">System Settings</h1>
            <p className="text-sm text-gray-500">Manage your application preferences</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={loadSettings}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button
              onClick={handleSubmit}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save Changes
                </>
              )}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-4 sticky top-4">
              <nav className="space-y-1">
                {[
                  { icon: SettingsIcon, label: 'General', id: 'general' },
                  { icon: Globe, label: 'Model', id: 'model' },
                  { icon: Bell, label: 'Notifications', id: 'notifications' },
                  { icon: Shield, label: 'Security', id: 'security' },
                  { icon: Database, label: 'Data', id: 'data' },
                ].map((item) => (
                  <button
                    key={item.id}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-sm rounded-lg hover:bg-gray-100 transition-colors text-gray-700"
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">General Settings</h2>
              
              <form className="space-y-6">
                {/* App Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Application Name
                  </label>
                  <input
                    type="text"
                    name="app_name"
                    value={settings.app_name || ''}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                </div>

                {/* Version */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Version
                  </label>
                  <input
                    type="text"
                    name="version"
                    value={settings.version || ''}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                </div>

                {/* Model Threshold */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Model Threshold
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      name="model_threshold"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={settings.model_threshold || 0.459}
                      onChange={handleChange}
                      className="flex-1"
                    />
                    <span className="text-sm font-medium text-gray-700 min-w-[40px]">
                      {(settings.model_threshold || 0.459).toFixed(2)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Risk score threshold for triggering alerts
                  </p>
                </div>

                {/* Prediction Window */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Prediction Window (hours)
                  </label>
                  <input
                    type="number"
                    name="prediction_window"
                    value={settings.prediction_window || 24}
                    onChange={handleChange}
                    min="1"
                    max="72"
                    className="w-32 px-3 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    How many hours ahead to predict
                  </p>
                </div>

                {/* Alert Frequency */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Alert Frequency (minutes)
                  </label>
                  <input
                    type="number"
                    name="alert_frequency"
                    value={settings.alert_frequency || 60}
                    onChange={handleChange}
                    min="5"
                    max="1440"
                    className="w-32 px-3 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    How often to check for new alerts
                  </p>
                </div>

                {/* Auto Prediction */}
                <div>
                  <label className="flex items-center gap-3 text-sm font-medium text-gray-700">
                    <input
                      type="checkbox"
                      name="auto_prediction"
                      checked={settings.auto_prediction || false}
                      onChange={handleChange}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    Enable Auto-Prediction
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    Automatically generate predictions for new patients
                  </p>
                </div>

                {/* Imputation Method */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Imputation Method
                  </label>
                  <select
                    name="imputation_method"
                    value={settings.imputation_method || 'knn'}
                    onChange={handleChange}
                    className="w-48 px-3 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  >
                    <option value="knn">K-Nearest Neighbors</option>
                    <option value="forward_fill">Forward Fill</option>
                    <option value="linear_interpolate">Linear Interpolate</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Method for handling missing data
                  </p>
                </div>

                {/* Email Notifications */}
                <div>
                  <label className="flex items-center gap-3 text-sm font-medium text-gray-700">
                    <input
                      type="checkbox"
                      name="email_notifications"
                      checked={settings.email_notifications || false}
                      onChange={handleChange}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    Email Notifications
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    Receive email alerts for critical predictions
                  </p>
                </div>

                {/* Data Retention */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Data Retention (days)
                  </label>
                  <input
                    type="number"
                    name="data_retention_days"
                    value={settings.data_retention_days || 30}
                    onChange={handleChange}
                    min="7"
                    max="365"
                    className="w-32 px-3 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    How long to keep patient data
                  </p>
                </div>

                {/* Maintenance Mode */}
                <div className="pt-4 border-t border-gray-200">
                  <label className="flex items-center gap-3 text-sm font-medium text-gray-700">
                    <input
                      type="checkbox"
                      name="maintenance_mode"
                      checked={settings.maintenance_mode || false}
                      onChange={handleChange}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    Maintenance Mode
                  </label>
                  <p className="text-xs text-red-500 mt-1">
                    Enable maintenance mode (only admins can access)
                  </p>
                </div>

                {/* Allow Registration */}
                <div>
                  <label className="flex items-center gap-3 text-sm font-medium text-gray-700">
                    <input
                      type="checkbox"
                      name="allow_registration"
                      checked={settings.allow_registration || false}
                      onChange={handleChange}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    Allow New Registrations
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    Allow new users to register
                  </p>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
