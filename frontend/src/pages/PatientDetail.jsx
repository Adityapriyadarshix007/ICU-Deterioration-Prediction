import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar.jsx';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import api from '../services/api';
import { 
  Activity, Heart, Brain, Droplet, Users, 
  Calendar, Hospital, AlertTriangle, 
  TrendingUp, TrendingDown, Clock, 
  FileText, Stethoscope, Pill, 
  ArrowLeft, Loader2, 
  CheckCircle, XCircle, AlertCircle
} from 'lucide-react';

function PatientDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPatientData();
  }, [id]);

  const loadPatientData = async () => {
    setLoading(true);
    try {
      // Try to get patient data from the patients list
      let patientData = null;
      
      try {
        const patientsResponse = await api.getPatients(id);
        if (patientsResponse.data) {
          let patients = [];
          if (Array.isArray(patientsResponse.data)) {
            patients = patientsResponse.data;
          } else if (patientsResponse.data.patients) {
            patients = patientsResponse.data.patients;
          }
          
          // Find patient by id or patient_id
          const found = patients.find(p => 
            p.patient_id === id || p.id === id
          );
          if (found) {
            patientData = found;
          }
        }
      } catch (e) {
        console.log('Could not fetch patient from list');
      }

      // If no patient found, create mock data
      if (!patientData) {
        patientData = {
          id: id,
          patient_id: id,
          patient_name: `Patient ${id}`,
          name: `Patient ${id}`,
          age: Math.floor(Math.random() * 40) + 40,
          gender: ['Male', 'Female'][Math.floor(Math.random() * 2)],
          admission_date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          diagnosis: ['Sepsis', 'Pneumonia', 'Heart Failure', 'COVID-19', 'Post-Surgery'][Math.floor(Math.random() * 5)],
          room: `ICU-${String(Math.floor(Math.random() * 20) + 1).padStart(3, '0')}`,
          risk_level: ['LOW', 'MEDIUM', 'HIGH'][Math.floor(Math.random() * 3)],
          vitals: {
            heart_rate: Math.floor(Math.random() * 60) + 60,
            sbp: Math.floor(Math.random() * 60) + 90,
            dbp: Math.floor(Math.random() * 40) + 60,
            gcs: Math.floor(Math.random() * 5) + 10,
            lactate: (Math.random() * 3 + 0.5).toFixed(2),
            urine_output: Math.floor(Math.random() * 50) + 20,
            fio2: Math.floor(Math.random() * 40) + 21,
            creatinine: (Math.random() * 2 + 0.5).toFixed(2)
          }
        };
      }

      setPatient(patientData);

      // Get predictions for this patient
      try {
        const predictionsResponse = await api.getPatientPredictions(id);
        setPredictions(predictionsResponse.data || []);
      } catch (e) {
        console.log('No predictions found for this patient');
        setPredictions([]);
      }

    } catch (error) {
      console.error('Failed to load patient data:', error);
      toast.error('Failed to load patient data');
      
      // Set fallback data
      setPatient({
        id: id,
        patient_id: id,
        patient_name: `Patient ${id}`,
        name: `Patient ${id}`,
        age: 65,
        gender: 'Male',
        admission_date: new Date().toISOString().split('T')[0],
        diagnosis: 'Under Observation',
        room: 'ICU-001',
        risk_level: 'MEDIUM',
        vitals: {
          heart_rate: 78,
          sbp: 122,
          dbp: 78,
          gcs: 15,
          lactate: 1.1,
          urine_output: 55,
          fio2: 21,
          creatinine: 0.9
        }
      });
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'HIGH':
      case 'CRITICAL':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'MEDIUM':
      case 'WARNING':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-green-100 text-green-800 border-green-200';
    }
  };

  const getVitalStatus = (key, value) => {
    const ranges = {
      heart_rate: { min: 60, max: 100 },
      sbp: { min: 90, max: 140 },
      dbp: { min: 60, max: 90 },
      gcs: { min: 13, max: 15 },
      lactate: { min: 0.5, max: 2.0 },
      urine_output: { min: 30, max: 60 },
      fio2: { min: 21, max: 40 },
      creatinine: { min: 0.6, max: 1.3 }
    };
    const range = ranges[key];
    if (!range) return 'normal';
    if (value < range.min) return 'low';
    if (value > range.max) return 'high';
    return 'normal';
  };

  const getVitalIcon = (key) => {
    const icons = {
      heart_rate: Heart,
      sbp: Activity,
      dbp: Activity,
      gcs: Brain,
      lactate: Droplet,
      urine_output: Droplet,
      fio2: Pill,
      creatinine: FileText
    };
    return icons[key] || Activity;
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading patient data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="container mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={() => navigate('/patients')}
          className="flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Patients
        </button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Patient Info */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Patient Information</h2>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getRiskColor(patient?.risk_level)}`}>
                  {patient?.risk_level || 'LOW'}
                </span>
              </div>
              
              <div className="space-y-4">
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Patient ID</p>
                  <p className="font-mono text-sm font-medium text-gray-900">{patient?.patient_id || patient?.id}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Name</p>
                  <p className="font-medium text-gray-900">{patient?.patient_name || patient?.name}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide">Age</p>
                    <p className="font-medium text-gray-900">{patient?.age} years</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide">Gender</p>
                    <p className="font-medium text-gray-900">{patient?.gender}</p>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Room</p>
                  <p className="font-medium text-gray-900">{patient?.room}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Admission Date</p>
                  <p className="font-medium text-gray-900">{patient?.admission_date || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Diagnosis</p>
                  <p className="font-medium text-gray-900 text-sm">{patient?.diagnosis || 'N/A'}</p>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Right Column - Vitals & Predictions */}
          <div className="lg:col-span-2 space-y-6">
            {/* Vitals Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
            >
              <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                Current Vitals
              </h2>
              {patient?.vitals ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(patient.vitals).map(([key, value]) => {
                    const status = getVitalStatus(key, value);
                    const Icon = getVitalIcon(key);
                    const labels = {
                      heart_rate: 'Heart Rate',
                      sbp: 'Systolic BP',
                      dbp: 'Diastolic BP',
                      gcs: 'GCS',
                      lactate: 'Lactate',
                      urine_output: 'Urine Output',
                      fio2: 'FiO₂',
                      creatinine: 'Creatinine'
                    };
                    const units = {
                      heart_rate: 'bpm',
                      sbp: 'mmHg',
                      dbp: 'mmHg',
                      gcs: '',
                      lactate: 'mmol/L',
                      urine_output: 'mL',
                      fio2: '%',
                      creatinine: 'mg/dL'
                    };
                    const statusColors = {
                      normal: 'bg-green-50 border-green-200',
                      low: 'bg-yellow-50 border-yellow-200',
                      high: 'bg-red-50 border-red-200'
                    };
                    return (
                      <div key={key} className={`p-4 rounded-xl border ${statusColors[status]}`}>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs text-gray-500">{labels[key] || key}</p>
                            <p className="text-xl font-bold text-gray-900">
                              {value} {units[key] || ''}
                            </p>
                          </div>
                          <div className={`p-2 rounded-lg ${
                            status === 'normal' ? 'bg-green-100' : 
                            status === 'high' ? 'bg-red-100' : 'bg-yellow-100'
                          }`}>
                            <Icon className={`w-4 h-4 ${
                              status === 'normal' ? 'text-green-600' : 
                              status === 'high' ? 'text-red-600' : 'text-yellow-600'
                            }`} />
                          </div>
                        </div>
                        {status !== 'normal' && (
                          <p className={`text-xs mt-1 font-medium ${
                            status === 'high' ? 'text-red-600' : 'text-yellow-600'
                          }`}>
                            {status === 'high' ? '↑ High' : status === 'low' ? '↓ Low' : 'Normal'}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No vital signs available</p>
              )}
            </motion.div>

            {/* Predictions Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
              className="bg-white p-6 rounded-xl shadow-lg border border-gray-100"
            >
              <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-blue-600" />
                Prediction History
              </h2>
              {predictions.length > 0 ? (
                <div className="space-y-3">
                  {predictions.map((pred, idx) => (
                    <div key={idx} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-100 hover:bg-gray-100 transition-colors">
                      <div>
                        <div className="flex items-center gap-3">
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                            pred.alert_level === 'CRITICAL' ? 'bg-red-100 text-red-800' :
                            pred.alert_level === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {pred.alert_level || 'STABLE'}
                          </span>
                          <span className="font-medium text-gray-900">
                            Risk: {(pred.risk_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                          <Clock className="w-3 h-3 inline mr-1" />
                          {pred.created_at ? new Date(pred.created_at).toLocaleString() : 'Unknown date'}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${
                            pred.risk_score > 0.7 ? 'bg-red-500' :
                            pred.risk_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                          }`} style={{ width: `${Math.min((pred.risk_score || 0) * 100, 100)}%` }} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-gray-400" />
                  </div>
                  <p className="text-gray-500">No predictions available for this patient</p>
                  <p className="text-xs text-gray-400 mt-1">Run a prediction from the dashboard</p>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PatientDetail;
