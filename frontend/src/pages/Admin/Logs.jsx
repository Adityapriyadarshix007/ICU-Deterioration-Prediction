import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';
import { FileText, Search, Filter, Clock, Activity, RefreshCw, Download } from 'lucide-react';

function Logs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    loadLogs();
  }, []);

  const loadLogs = async () => {
    setLoading(true);
    try {
      const response = await api.getActivityLogs();
      setLogs(response.data.logs || []);
    } catch (error) {
      console.error('Failed to load logs:', error);
      toast.error('Failed to load logs');
      if (error.response?.status === 403) {
        navigate('/dashboard');
      }
      // Set demo logs
      setLogs([
        { id: 1, user: 'admin', action: 'Login', patient_name: 'N/A', alert_level: 'INFO', created_at: new Date(Date.now() - 1000 * 60 * 5).toISOString() },
        { id: 2, user: 'doctor', action: 'Prediction', patient_name: 'John Doe', alert_level: 'CRITICAL', created_at: new Date(Date.now() - 1000 * 60 * 15).toISOString() },
        { id: 3, user: 'doctor', action: 'Prediction', patient_name: 'Jane Smith', alert_level: 'HIGH', created_at: new Date(Date.now() - 1000 * 60 * 30).toISOString() },
        { id: 4, user: 'admin', action: 'User Management', patient_name: 'N/A', alert_level: 'INFO', created_at: new Date(Date.now() - 1000 * 60 * 60).toISOString() },
        { id: 5, user: 'doctor', action: 'Patient Update', patient_name: 'Robert Johnson', alert_level: 'MEDIUM', created_at: new Date(Date.now() - 1000 * 60 * 90).toISOString() }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const filteredLogs = logs.filter(log => {
    const search = searchTerm.toLowerCase();
    const matchesSearch = 
      (log.user || '').toLowerCase().includes(search) ||
      (log.patient_name || '').toLowerCase().includes(search) ||
      (log.action || '').toLowerCase().includes(search);
    
    if (filter === 'all') return matchesSearch;
    if (filter === 'critical') return matchesSearch && (log.alert_level === 'CRITICAL' || log.alert_level === 'HIGH');
    if (filter === 'info') return matchesSearch && (log.alert_level === 'INFO' || log.alert_level === 'MEDIUM');
    return matchesSearch;
  });

  const getLogColor = (level) => {
    const map = {
      'CRITICAL': 'bg-red-100 text-red-800',
      'HIGH': 'bg-orange-100 text-orange-800',
      'MEDIUM': 'bg-yellow-100 text-yellow-800',
      'INFO': 'bg-blue-100 text-blue-800',
      'WARNING': 'bg-yellow-100 text-yellow-800'
    };
    return map[level] || 'bg-gray-100 text-gray-800';
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
            <h1 className="text-2xl font-bold">Activity Logs</h1>
            <p className="text-sm text-gray-500">System activity and audit trail</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={loadLogs}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
              <Download className="w-4 h-4" />
              Export Logs
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white p-4 rounded-xl shadow mb-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setFilter('all')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilter('critical')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'critical' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Critical
              </button>
              <button
                onClick={() => setFilter('info')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === 'info' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Info
              </button>
            </div>
          </div>
        </div>

        {/* Logs Table */}
        <div className="bg-white rounded-xl shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Level</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredLogs.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-8 text-center text-gray-500">
                      No logs found
                    </td>
                  </tr>
                ) : (
                  filteredLogs.map((log, index) => (
                    <tr key={log.id || index} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 font-medium text-gray-900">{log.user || 'System'}</td>
                      <td className="px-6 py-4 text-gray-600">{log.action || 'Unknown'}</td>
                      <td className="px-6 py-4 text-gray-600">{log.patient_name || log.patient_id || 'N/A'}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getLogColor(log.alert_level)}`}>
                          {log.alert_level || 'INFO'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-4 text-sm text-gray-500">
          Total Logs: {filteredLogs.length}
        </div>
      </div>
    </div>
  );
}

export default Logs;
