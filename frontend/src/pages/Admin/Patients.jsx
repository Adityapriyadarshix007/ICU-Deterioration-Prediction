import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';
import { X, AlertTriangle, Trash2 } from 'lucide-react';

function Patients() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [patientToDelete, setPatientToDelete] = useState(null);
  const [editingPatient, setEditingPatient] = useState(null);
  const [formData, setFormData] = useState({
    patient_name: '',
    age: '',
    gender: '',
    diagnosis: '',
    room: '',
    risk_level: 'LOW'
  });
  const [importing, setImporting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadPatients();
  }, []);

  const loadPatients = async () => {
    setLoading(true);
    try {
      const response = await api.getAdminPatients(searchTerm);
      console.log('Admin patients response:', response.data);
      
      let patientData = [];
      if (response.data) {
        if (Array.isArray(response.data)) {
          patientData = response.data;
        } else if (response.data.patients) {
          patientData = response.data.patients;
        }
      }
      
      setPatients(patientData);
    } catch (error) {
      console.error('Failed to load patients:', error);
      if (error.response?.status === 403) {
        toast.error('Admin access required');
        navigate('/dashboard');
      } else {
        toast.error('Failed to load patients');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    loadPatients();
  };

  const handleAddPatient = async (e) => {
    e.preventDefault();
    try {
      const patientData = {
        patient_name: formData.patient_name,
        age: parseInt(formData.age) || 0,
        gender: formData.gender,
        diagnosis: formData.diagnosis,
        room: formData.room,
        risk_level: formData.risk_level,
        status: 'Active'
      };
      
      console.log('Creating patient with data:', patientData);
      await api.createPatient(patientData);
      toast.success('Patient added successfully!');
      setShowAddModal(false);
      setFormData({ patient_name: '', age: '', gender: '', diagnosis: '', room: '', risk_level: 'LOW' });
      loadPatients();
    } catch (error) {
      console.error('Failed to add patient:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to add patient';
      toast.error(errorMsg);
    }
  };

  const handleUpdatePatient = async (e) => {
    e.preventDefault();
    try {
      const patientId = editingPatient.patient_id || editingPatient.id;
      await api.updatePatient(patientId, formData);
      toast.success('Patient updated successfully!');
      setEditingPatient(null);
      setFormData({ patient_name: '', age: '', gender: '', diagnosis: '', room: '', risk_level: 'LOW' });
      loadPatients();
    } catch (error) {
      console.error('Failed to update patient:', error);
      toast.error('Failed to update patient');
    }
  };

  const handleDeleteClick = (patient) => {
    setPatientToDelete(patient);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async () => {
    if (!patientToDelete) return;
    
    try {
      const patientId = patientToDelete.id || patientToDelete.patient_id;
      await api.deletePatient(patientId);
      toast.success('Patient deleted successfully!');
      setShowDeleteModal(false);
      setPatientToDelete(null);
      loadPatients();
    } catch (error) {
      console.error('Failed to delete patient:', error);
      toast.error('Failed to delete patient');
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
    setPatientToDelete(null);
  };

  const handleImportCSV = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setImporting(true);
    try {
      await api.importPatients(file);
      toast.success('Patients imported successfully!');
      loadPatients();
    } catch (error) {
      console.error('Failed to import patients:', error);
      toast.error('Failed to import patients');
    } finally {
      setImporting(false);
      e.target.value = '';
    }
  };

  const handleExportCSV = async () => {
    setExporting(true);
    try {
      const response = await api.exportPatients();
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'patients_export.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Patients exported successfully!');
    } catch (error) {
      console.error('Failed to export patients:', error);
      toast.error('Failed to export patients');
    } finally {
      setExporting(false);
    }
  };

  const openEditModal = (patient) => {
    setEditingPatient(patient);
    setFormData({
      patient_name: patient.patient_name || '',
      age: patient.age || '',
      gender: patient.gender || '',
      diagnosis: patient.diagnosis || '',
      room: patient.room || '',
      risk_level: patient.risk_level || 'LOW'
    });
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'HIGH':
      case 'CRITICAL':
        return 'bg-red-100 text-red-800';
      case 'MEDIUM':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
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
            <h1 className="text-2xl font-bold">Patient Management</h1>
            <p className="text-sm text-gray-500">Manage patient records with import/export</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={() => setShowAddModal(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
            >
              + Add Patient
            </button>
            <label className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition cursor-pointer">
              {importing ? 'Importing...' : '📥 Import CSV'}
              <input
                type="file"
                accept=".csv"
                onChange={handleImportCSV}
                className="hidden"
                disabled={importing}
              />
            </label>
            <button
              onClick={handleExportCSV}
              disabled={exporting}
              className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition disabled:opacity-50"
            >
              {exporting ? 'Exporting...' : '📤 Export CSV'}
            </button>
          </div>
        </div>

        <div className="mb-6 flex gap-3">
          <input
            type="text"
            placeholder="Search patients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
          />
          <button
            onClick={handleSearch}
            className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition"
          >
            Search
          </button>
          <button
            onClick={() => {
              setSearchTerm('');
              loadPatients();
            }}
            className="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition"
          >
            Reset
          </button>
        </div>

        <div className="bg-white rounded-xl shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Age</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gender</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Room</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {patients.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="px-6 py-8 text-center text-gray-500">
                      No patients found. Add your first patient or import from CSV.
                    </td>
                  </tr>
                ) : (
                  patients.map((patient) => (
                    <tr key={patient.id || patient.patient_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">
                        {patient.patient_id || patient.id}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        {patient.patient_name || patient.name}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">{patient.age || 'N/A'}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{patient.gender || 'N/A'}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{patient.room || 'N/A'}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getRiskColor(patient.risk_level)}`}>
                          {patient.risk_level || 'LOW'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => openEditModal(patient)}
                            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => handleDeleteClick(patient)}
                            className="text-red-600 hover:text-red-800 text-sm font-medium"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && patientToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-red-100 rounded-full">
                <Trash2 className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">Delete Patient</h2>
                <p className="text-sm text-gray-500">This action cannot be undone</p>
              </div>
            </div>
            
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-red-800 font-medium">Are you sure you want to delete this patient?</p>
                  <p className="text-sm text-red-600 mt-1">
                    <strong>Patient:</strong> {patientToDelete.patient_name || patientToDelete.name}
                    <br />
                    <strong>ID:</strong> {patientToDelete.patient_id || patientToDelete.id}
                  </p>
                  <p className="text-xs text-red-500 mt-2">
                    All associated predictions and data will be permanently removed.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelDelete}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete Patient
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add/Edit Modal */}
      {(showAddModal || editingPatient) && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">
                {editingPatient ? 'Edit Patient' : 'Add New Patient'}
              </h2>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setEditingPatient(null);
                  setFormData({ patient_name: '', age: '', gender: '', diagnosis: '', room: '', risk_level: 'LOW' });
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={editingPatient ? handleUpdatePatient : handleAddPatient}>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Patient Name *</label>
                  <input
                    type="text"
                    value={formData.patient_name}
                    onChange={(e) => setFormData({...formData, patient_name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    required
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Age</label>
                    <input
                      type="number"
                      value={formData.age}
                      onChange={(e) => setFormData({...formData, age: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                      min={0}
                      max={150}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Gender</label>
                    <select
                      value={formData.gender}
                      onChange={(e) => setFormData({...formData, gender: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    >
                      <option value="">Select</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Diagnosis</label>
                  <input
                    type="text"
                    value={formData.diagnosis}
                    onChange={(e) => setFormData({...formData, diagnosis: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Room</label>
                    <input
                      type="text"
                      value={formData.room}
                      onChange={(e) => setFormData({...formData, room: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Risk Level</label>
                    <select
                      value={formData.risk_level}
                      onChange={(e) => setFormData({...formData, risk_level: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    >
                      <option value="LOW">Low</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="HIGH">High</option>
                      <option value="CRITICAL">Critical</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  type="button"
                  onClick={() => {
                    setShowAddModal(false);
                    setEditingPatient(null);
                    setFormData({ patient_name: '', age: '', gender: '', diagnosis: '', room: '', risk_level: 'LOW' });
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  {editingPatient ? 'Update' : 'Add'} Patient
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default Patients;
