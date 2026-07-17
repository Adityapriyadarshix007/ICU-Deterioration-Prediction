import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';

function AdminDashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    try {
      const response = await api.getOverview();
      setStats(response.data);
    } catch (error) {
      toast.error('Failed to load admin dashboard');
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
        <Breadcrumb />
        <h1 className="text-2xl font-bold mb-6">Admin Dashboard</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-gray-500 text-sm">Total Users</p>
            <p className="text-3xl font-bold text-blue-600">{stats?.users?.total || 0}</p>
            <p className="text-xs text-gray-400">Active: {stats?.users?.active || 0}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-gray-500 text-sm">Total Predictions</p>
            <p className="text-3xl font-bold text-green-600">{stats?.predictions?.total || 0}</p>
            <p className="text-xs text-gray-400">Weekly: {stats?.predictions?.weekly || 0}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-gray-500 text-sm">High Risk Patients</p>
            <p className="text-3xl font-bold text-red-600">{stats?.predictions?.high_risk || 0}</p>
            <p className="text-xs text-gray-400">Critical: {stats?.predictions?.critical || 0}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-gray-500 text-sm">Avg Risk Score</p>
            <p className="text-3xl font-bold text-purple-600">
              {((stats?.predictions?.avg_risk || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition">
            <h3 className="font-semibold mb-2">👤 Patients</h3>
            <p className="text-sm text-gray-600">Manage patient records</p>
            <button
              onClick={() => navigate('/admin/patients')}
              className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Manage Patients →
            </button>
          </div>
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition">
            <h3 className="font-semibold mb-2">👥 Users</h3>
            <p className="text-sm text-gray-600">Manage users and roles</p>
            <button
              onClick={() => navigate('/admin/users')}
              className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Manage Users →
            </button>
          </div>
          <div className="bg-white p-6 rounded-xl shadow hover:shadow-lg transition">
            <h3 className="font-semibold mb-2">⚙️ Settings</h3>
            <p className="text-sm text-gray-600">Configure system settings</p>
            <button
              onClick={() => navigate('/admin/settings')}
              className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Configure →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminDashboard;
