import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Search, Users, Activity, AlertTriangle, 
  Plus, Filter, Download, Eye, 
  Loader2, User, Calendar, Hospital,
  Stethoscope, Clock, ArrowRight, RefreshCw,
  X, Check, Trash2, Edit2, Eye as EyeIcon
} from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar.jsx';
import Card from '../components/ui/Card.jsx';
import Button from '../components/ui/Button.jsx';
import StatusBadge from '../components/ui/StatusBadge.jsx';
import api from '../services/api';

function PatientSearch() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('all');
  const [totalPatients, setTotalPatients] = useState(0);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingPatient, setEditingPatient] = useState(null);
  const [formData, setFormData] = useState({
    patient_name: '',
    age: '',
    gender: '',
    diagnosis: '',
    room: '',
    risk_level: 'LOW'
  });

  useEffect(() => {
    loadPatients();
  }, []);

  const loadPatients = async () => {
    setLoading(true);
    try {
      console.log('🔍 Fetching patients with search:', searchTerm);
      
      const response = await api.getPatients(searchTerm);
      console.log('📊 Patients API response:', response.data);
      
      let patientData = [];
      let total = 0;
      
      if (response.data) {
        if (Array.isArray(response.data)) {
          patientData = response.data;
          total = patientData.length;
        } else if (response.data.patients) {
          patientData = response.data.patients;
          total = response.data.total || patientData.length;
        } else {
          patientData = Object.values(response.data).flat().filter(item => 
            item && typeof item === 'object' && (item.patient_id || item.id)
          );
          total = patientData.length;
        }
      }
      
      console.log(`✅ Found ${patientData.length} patients`);
      console.log('📋 Patient risk levels:', patientData.map(p => ({ 
        name: p.patient_name || p.name, 
        risk: p.risk_level 
      })));
      
      setPatients(patientData);
      setTotalPatients(total);
      
    } catch (error) {
      console.error('❌ Failed to load patients:', error);
      
      if (error.response?.status === 401) {
        toast.error('Please login to view patients');
        navigate('/login');
      } else {
        toast.error('Failed to load patients from database');
        setPatients([]);
        setTotalPatients(0);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    loadPatients();
  };

  const handleFilterChange = (newFilter) => {
    setFilter(newFilter);
  };

  const handleRefresh = () => {
    loadPatients();
    toast.success('Refreshed patient list');
  };

  const handleViewPatient = (patient) => {
    const patientId = patient.patient_id || patient.id;
    if (patientId) {
      navigate(`/patients/${patientId}`);
    } else {
      toast.error('Patient ID not found');
    }
  };

  const handleAddPatient = async (e) => {
    e.preventDefault();
    try {
      if (!formData.patient_name || !formData.age || !formData.gender || !formData.room || !formData.diagnosis) {
        toast.error('Please fill in all required fields');
        return;
      }

      const newPatient = {
        patient_id: `PAT-${Date.now().toString().slice(-8)}`,
        patient_name: formData.patient_name,
        age: parseInt(formData.age),
        gender: formData.gender,
        room: formData.room,
        diagnosis: formData.diagnosis,
        risk_level: formData.risk_level,
        status: 'Active'
      };

      await api.createPatient(newPatient);
      toast.success('Patient added successfully!');
      setShowAddModal(false);
      resetForm();
      loadPatients();
    } catch (error) {
      console.error('Failed to add patient:', error);
      toast.error('Failed to add patient');
    }
  };

  const handleEditPatient = (patient) => {
    setEditingPatient(patient);
    setFormData({
      patient_name: patient.patient_name || '',
      age: patient.age || '',
      gender: patient.gender || '',
      diagnosis: patient.diagnosis || '',
      room: patient.room || '',
      risk_level: patient.risk_level || 'LOW'
    });
    setShowAddModal(true);
  };

  const handleUpdatePatient = async (e) => {
    e.preventDefault();
    try {
      const patientId = editingPatient.patient_id || editingPatient.id;
      await api.updatePatient(patientId, formData);
      toast.success('Patient updated successfully!');
      setShowAddModal(false);
      setEditingPatient(null);
      resetForm();
      loadPatients();
    } catch (error) {
      console.error('Failed to update patient:', error);
      toast.error('Failed to update patient');
    }
  };

  const handleDeletePatient = async (patient) => {
    if (!window.confirm(`Are you sure you want to delete ${patient.patient_name || patient.name}?`)) return;
    
    try {
      const patientId = patient.patient_id || patient.id;
      await api.deletePatient(patientId);
      toast.success('Patient deleted successfully!');
      loadPatients();
    } catch (error) {
      console.error('Failed to delete patient:', error);
      toast.error('Failed to delete patient');
    }
  };

  const resetForm = () => {
    setFormData({
      patient_name: '',
      age: '',
      gender: '',
      diagnosis: '',
      room: '',
      risk_level: 'LOW'
    });
    setEditingPatient(null);
  };

  const handleExportCSV = async () => {
    try {
      toast.loading('Exporting patients...', { id: 'export' });
      
      const headers = ['Patient ID', 'Name', 'Age', 'Gender', 'Room', 'Diagnosis', 'Risk Level', 'Status'];
      const rows = patients.map(p => [
        p.patient_id || p.id,
        p.patient_name || p.name,
        p.age || 'N/A',
        p.gender || 'N/A',
        p.room || 'N/A',
        p.diagnosis || 'N/A',
        p.risk_level || 'LOW',
        p.status || 'Active'
      ]);
      
      const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `patients_${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      toast.dismiss('export');
      toast.success(`Exported ${patients.length} patients`);
    } catch (error) {
      console.error('Failed to export patients:', error);
      toast.dismiss('export');
      toast.error('Failed to export patients');
    }
  };

  // Filter patients based on exact risk levels from database
  const filteredPatients = patients.filter(patient => {
    // Search filter
    const name = (patient.patient_name || patient.name || '').toLowerCase();
    const search = searchTerm.toLowerCase();
    const matchesSearch = name.includes(search) || 
                         (patient.patient_id || patient.id || '').toLowerCase().includes(search);
    
    if (!matchesSearch) return false;
    
    // Get risk level (exact match from database)
    const riskLevel = (patient.risk_level || 'LOW').toUpperCase();
    
    // Filter logic based on exact database values
    if (filter === 'all') return true;
    
    if (filter === 'low') {
      return riskLevel === 'LOW';
    }
    
    if (filter === 'medium') {
      return riskLevel === 'MEDIUM';
    }
    
    if (filter === 'high') {
      return riskLevel === 'HIGH';
    }
    
    if (filter === 'critical') {
      return riskLevel === 'CRITICAL';
    }
    
    return true;
  });

  // Count patients by risk level
  const getRiskCount = (level) => {
    return patients.filter(p => {
      const risk = (p.risk_level || 'LOW').toUpperCase();
      return risk === level;
    }).length;
  };

  const getRiskBadge = (risk) => {
    const riskLevel = (risk || '').toUpperCase();
    const map = {
      'CRITICAL': 'critical',
      'HIGH': 'critical',
      'MEDIUM': 'warning',
      'LOW': 'stable'
    };
    return map[riskLevel] || 'stable';
  };

  const getRiskLabel = (risk) => {
    const riskLevel = (risk || '').toUpperCase();
    const map = {
      'CRITICAL': 'CRITICAL',
      'HIGH': 'HIGH',
      'MEDIUM': 'MEDIUM',
      'LOW': 'LOW'
    };
    return map[riskLevel] || 'LOW';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Loading patients...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Patient Search</h1>
            <p className="text-gray-500">Find and manage patient records</p>
            {patients.length === 0 && (
              <p className="text-xs text-yellow-600 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                No patients found. Add your first patient!
              </p>
            )}
          </div>
          <div className="flex items-center gap-3 mt-4 md:mt-0">
            <Button variant="outline" size="sm" icon={RefreshCw} onClick={handleRefresh}>
              Refresh
            </Button>
            <Button variant="outline" size="sm" icon={Download} onClick={handleExportCSV}>
              Export CSV
            </Button>
            <Button size="sm" icon={Plus} onClick={() => {
              resetForm();
              setShowAddModal(true);
            }}>
              Add Patient
            </Button>
          </div>
        </div>

        {/* Search & Filter - Updated to match database levels */}
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 mb-6">
          <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by patient name or ID..."
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => handleFilterChange('all')}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                All ({patients.length})
              </button>
              <button
                type="button"
                onClick={() => handleFilterChange('low')}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  filter === 'low' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                LOW ({getRiskCount('LOW')})
              </button>
              <button
                type="button"
                onClick={() => handleFilterChange('medium')}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  filter === 'medium' ? 'bg-yellow-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                MEDIUM ({getRiskCount('MEDIUM')})
              </button>
              <button
                type="button"
                onClick={() => handleFilterChange('high')}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  filter === 'high' ? 'bg-orange-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                HIGH ({getRiskCount('HIGH')})
              </button>
              <button
                type="button"
                onClick={() => handleFilterChange('critical')}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  filter === 'critical' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                CRITICAL ({getRiskCount('CRITICAL')})
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Search
              </button>
            </div>
          </form>
        </div>

        {/* Patient List - Clickable Rows */}
        <Card title={`Patients (${filteredPatients.length})`} icon={Users}>
          {filteredPatients.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="w-8 h-8 text-gray-400" />
              </div>
              <p className="text-gray-500">No patients found</p>
              <p className="text-sm text-gray-400">
                {searchTerm ? 'Try adjusting your search' : 'Add patients from the admin panel'}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Patient ID</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Name</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Age</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Gender</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Room</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Risk</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map((patient, index) => (
                    <motion.tr
                      key={patient.patient_id || patient.id || index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-b border-gray-100 hover:bg-blue-50/50 transition-colors cursor-pointer group"
                      onClick={() => handleViewPatient(patient)}
                    >
                      <td className="py-3 px-4 text-sm font-mono text-gray-600">
                        {patient.patient_id || patient.id || 'N/A'}
                      </td>
                      <td className="py-3 px-4 font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                        {patient.patient_name || patient.name || 'Unknown'}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">{patient.age || 'N/A'}</td>
                      <td className="py-3 px-4 text-sm text-gray-600">{patient.gender || 'N/A'}</td>
                      <td className="py-3 px-4 text-sm text-gray-600">{patient.room || 'N/A'}</td>
                      <td className="py-3 px-4">
                        <StatusBadge 
                          status={getRiskBadge(patient.risk_level)}
                          label={getRiskLabel(patient.risk_level)}
                          size="sm"
                        />
                      </td>
                      <td className="py-3 px-4">
                        <StatusBadge 
                          status={patient.status === 'Active' ? 'stable' : patient.status === 'Critical' ? 'critical' : 'warning'}
                          label={patient.status || 'Active'}
                          size="sm"
                        />
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => handleViewPatient(patient)}
                            className="p-1.5 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                            title="View Patient Details"
                          >
                            <EyeIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleEditPatient(patient)}
                            className="p-1.5 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                            title="Edit Patient"
                          >
                            <Edit2 className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDeletePatient(patient)}
                            className="p-1.5 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                            title="Delete Patient"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* Add/Edit Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl p-6 w-full max-w-md max-h-[90vh] overflow-y-auto"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">
                {editingPatient ? 'Edit Patient' : 'Add New Patient'}
              </h2>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  resetForm();
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <form onSubmit={editingPatient ? handleUpdatePatient : handleAddPatient}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Patient Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={formData.patient_name}
                    onChange={(e) => setFormData({...formData, patient_name: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                    required
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Age <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      value={formData.age}
                      onChange={(e) => setFormData({...formData, age: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      min={18}
                      max={120}
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Gender <span className="text-red-500">*</span>
                    </label>
                    <select
                      value={formData.gender}
                      onChange={(e) => setFormData({...formData, gender: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      required
                    >
                      <option value="">Select</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Diagnosis <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={formData.diagnosis}
                    onChange={(e) => setFormData({...formData, diagnosis: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                    required
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Room <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      value={formData.room}
                      onChange={(e) => setFormData({...formData, room: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Risk Level
                    </label>
                    <select
                      value={formData.risk_level}
                      onChange={(e) => setFormData({...formData, risk_level: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                    >
                      <option value="LOW">Low</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="HIGH">High</option>
                      <option value="CRITICAL">Critical</option>
                    </select>
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => {
                    setShowAddModal(false);
                    resetForm();
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-6 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
                >
                  {editingPatient ? 'Update' : 'Add'} Patient
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
}

export default PatientSearch;
