import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';
import { 
  BarChart3, TrendingUp, Users, Activity, Brain, 
  Clock, Download, RefreshCw, AlertTriangle 
} from 'lucide-react';

function Analytics() {
  const [analytics, setAnalytics] = useState({
    total_predictions: 0,
    high_risk_patients: 0,
    critical_alerts: 0,
    avg_risk_score: 0,
    predictions_by_day: [],
    risk_distribution: { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 },
    users: { total: 0, active: 0, admins: 0 }
  });
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const response = await api.getOverview();
      const data = response.data || {};
      
      console.log('📊 Analytics data received:', data);
      
      // ============================================================
      // CRITICAL FIX: Read the correct structure from backend
      // Backend returns: { users: {...}, predictions: {...}, risk_distribution: {...}, trends: [...] }
      // ============================================================
      const predictions = data.predictions || {};
      const riskDistribution = data.risk_distribution || { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };
      const trends = data.trends || [];
      const users = data.users || { total: 0, active: 0, admins: 0 };
      
      // Ensure risk_distribution has all required keys
      const safeRiskDistribution = {
        LOW: riskDistribution.LOW || 0,
        MEDIUM: riskDistribution.MEDIUM || 0,
        HIGH: riskDistribution.HIGH || 0,
        CRITICAL: riskDistribution.CRITICAL || 0
      };
      
      setAnalytics({
        total_predictions: predictions.total || 0,
        high_risk_patients: predictions.high_risk || 0,
        critical_alerts: predictions.critical || 0,
        avg_risk_score: predictions.avg_risk || 0,
        predictions_by_day: trends,
        risk_distribution: safeRiskDistribution,
        users: users
      });
      
    } catch (error) {
      console.error('❌ Failed to load analytics:', error);
      toast.error('Failed to load analytics');
      
      if (error.response?.status === 403) {
        navigate('/dashboard');
      }
      
      // Set mock data for demo
      setAnalytics({
        total_predictions: 156,
        high_risk_patients: 23,
        critical_alerts: 8,
        avg_risk_score: 0.42,
        predictions_by_day: [
          { date: '2026-07-14', count: 12 },
          { date: '2026-07-15', count: 18 },
          { date: '2026-07-16', count: 15 },
          { date: '2026-07-17', count: 22 },
          { date: '2026-07-18', count: 19 },
          { date: '2026-07-19', count: 25 },
          { date: '2026-07-20', count: 20 }
        ],
        risk_distribution: { LOW: 45, MEDIUM: 35, HIGH: 15, CRITICAL: 5 },
        users: { total: 12, active: 8, admins: 2 }
      });
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

  const riskDist = analytics.risk_distribution || { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };
  const totalPatients = Object.values(riskDist).reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-screen bg-gray-100">
      <AdminSidebar />
      <div className="ml-64 p-8">
        <Breadcrumb />
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
            <p className="text-sm text-gray-500">System performance and usage metrics</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={loadAnalytics}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
              <Download className="w-4 h-4" />
              Export Report
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Total Predictions</p>
                <p className="text-3xl font-bold text-blue-600">{analytics.total_predictions}</p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <Activity className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">All time</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">High Risk Patients</p>
                <p className="text-3xl font-bold text-red-600">{analytics.high_risk_patients}</p>
              </div>
              <div className="p-3 bg-red-100 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">Active high risk cases</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Critical Alerts</p>
                <p className="text-3xl font-bold text-orange-600">{analytics.critical_alerts}</p>
              </div>
              <div className="p-3 bg-orange-100 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-orange-600" />
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">Requires immediate attention</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Avg Risk Score</p>
                <p className="text-3xl font-bold text-purple-600">{(analytics.avg_risk_score * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-purple-100 rounded-lg">
                <Brain className="w-6 h-6 text-purple-600" />
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">Across all patients</p>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl shadow">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users className="w-5 h-5 text-gray-500" />
              Risk Distribution
            </h3>
            <div className="space-y-4">
              {Object.entries(riskDist).map(([level, count]) => {
                const percentage = totalPatients > 0 ? (count / totalPatients) * 100 : 0;
                const colors = {
                  LOW: 'bg-green-500',
                  MEDIUM: 'bg-yellow-500',
                  HIGH: 'bg-orange-500',
                  CRITICAL: 'bg-red-500'
                };
                return (
                  <div key={level}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-gray-700">{level}</span>
                      <span className="text-gray-500">{count} patients ({percentage.toFixed(1)}%)</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div className={`${colors[level] || 'bg-gray-400'} h-2.5 rounded-full transition-all duration-500`} style={{ width: `${Math.min(percentage, 100)}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white p-6 rounded-xl shadow">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-500" />
              Recent Activity
            </h3>
            <div className="space-y-3">
              {analytics.predictions_by_day && analytics.predictions_by_day.length > 0 ? (
                analytics.predictions_by_day.slice(-5).map((day, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">{day.date}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-blue-600">{day.count}</span>
                      <span className="text-xs text-gray-400">predictions</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>No recent activity</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* User Stats */}
        <div className="mt-6 bg-white p-6 rounded-xl shadow">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-gray-500" />
            User Statistics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-2xl font-bold text-blue-600">{analytics.users?.total || 0}</p>
              <p className="text-sm text-gray-600">Total Users</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-2xl font-bold text-green-600">{analytics.users?.active || 0}</p>
              <p className="text-sm text-gray-600">Active Users</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-2xl font-bold text-purple-600">{analytics.users?.admins || 0}</p>
              <p className="text-sm text-gray-600">Administrators</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics;