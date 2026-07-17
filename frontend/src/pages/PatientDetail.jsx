import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar.jsx';
import toast from 'react-hot-toast';
import api from '../services/api';

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
    try {
      setPatient({
        id: id,
        name: 'John Doe',
        age: 65,
        gender: 'Male',
        admissionDate: '2026-07-14',
        diagnosis: 'Sepsis with Multi-Organ Dysfunction',
        room: 'ICU-101',
        vitals: {
          heart_rate: 95,
          sbp: 105,
          dbp: 65,
          gcs: 11,
          lactate: 2.5,
          urine_output: 35,
          fio2: 45,
          creatinine: 1.8
        }
      });
      const response = await api.getPatientPredictions(id);
      setPredictions(response.data || []);
    } catch (error) {
      toast.error('Failed to load patient data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="container mx-auto px-4 py-8">
        <button
          onClick={() => navigate('/patients')}
          className="text-primary-600 hover:text-primary-800 mb-4 inline-block"
        >
          ← Back to Patients
        </button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-bold mb-4">Patient Information</h2>
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-500">Patient ID</p>
                  <p className="font-medium">{patient?.id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Name</p>
                  <p className="font-medium">{patient?.name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Age / Gender</p>
                  <p className="font-medium">{patient?.age} / {patient?.gender}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Room</p>
                  <p className="font-medium">{patient?.room}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Admission Date</p>
                  <p className="font-medium">{patient?.admissionDate}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Diagnosis</p>
                  <p className="font-medium text-sm">{patient?.diagnosis}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="bg-white p-6 rounded-xl shadow-lg mb-6">
              <h2 className="text-xl font-bold mb-4">Current Vitals</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {patient?.vitals && Object.entries(patient.vitals).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-xs text-gray-500 capitalize">{key.replace('_', ' ')}</p>
                    <p className="text-lg font-bold text-primary-600">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-bold mb-4">Prediction History</h2>
              {predictions.length > 0 ? (
                <div className="space-y-3">
                  {predictions.map((pred, idx) => (
                    <div key={idx} className="border-l-4 border-primary-500 pl-4 py-2">
                      <div className="flex justify-between">
                        <span className="font-medium">Risk Score: {(pred.risk_score * 100).toFixed(1)}%</span>
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                          pred.alert_level === 'CRITICAL' ? 'bg-red-100 text-red-800' :
                          pred.alert_level === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {pred.alert_level}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500">
                        {new Date(pred.created_at).toLocaleString()}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No predictions yet</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PatientDetail;
