import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar.jsx';
import api from '../services/api';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [patientData, setPatientData] = useState({
    patient_name: '',
    heart_rate: '',
    sbp: '',
    dbp: '',
    gcs: '',
    lactate: '',
    urine_output: '',
    fio2: '',
    creatinine: ''
  });

  // Auto-generate Patient ID
  const [patientId, setPatientId] = useState('');

  useEffect(() => {
    loadStats();
    generatePatientId();
  }, []);

  const generatePatientId = () => {
    const date = new Date();
    const dateStr = date.toISOString().slice(0,10).replace(/-/g, '');
    const timeStr = date.toTimeString().slice(0,8).replace(/:/g, '');
    const random = Math.floor(Math.random() * 1000).toString().padStart(3, '0');
    setPatientId(`ICU-${dateStr}-${timeStr}-${random}`);
  };

  const loadStats = async () => {
    try {
      const response = await api.getDashboardStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load dashboard stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setPatientData({ ...patientData, [e.target.name]: e.target.value });
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    // Validate inputs
    const requiredFields = ['patient_name', 'heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine'];
    const missing = requiredFields.filter(field => !patientData[field] || patientData[field] === '');

    if (missing.length > 0) {
      toast.error(`Please fill in: ${missing.map(f => f.replace('_', ' ').toUpperCase()).join(', ')}`);
      return;
    }

    setPredicting(true);
    try {
      // Generate a new patient ID for this prediction
      generatePatientId();

      const predictionData = {
        patient_id: patientId,
        patient_name: patientData.patient_name,
        heart_rate: parseFloat(patientData.heart_rate),
        sbp: parseFloat(patientData.sbp),
        dbp: parseFloat(patientData.dbp),
        gcs: parseFloat(patientData.gcs),
        lactate: parseFloat(patientData.lactate),
        urine_output: parseFloat(patientData.urine_output),
        fio2: parseFloat(patientData.fio2),
        creatinine: parseFloat(patientData.creatinine)
      };

      const response = await api.predict(predictionData);
      setPrediction(response.data);
      toast.success('✅ Prediction complete!');
    } catch (error) {
      toast.error('Prediction failed. Please try again.');
      console.error('Prediction error:', error);
    } finally {
      setPredicting(false);
    }
  };

  const resetForm = () => {
    setPatientData({
      patient_name: '',
      heart_rate: '',
      sbp: '',
      dbp: '',
      gcs: '',
      lactate: '',
      urine_output: '',
      fio2: '',
      creatinine: ''
    });
    setPrediction(null);
    generatePatientId();
    toast.info('Form reset. Ready for new patient.');
  };

  const getAlertColor = (level) => {
    switch(level) {
      case 'CRITICAL': return 'bg-red-600';
      case 'WARNING': return 'bg-yellow-600';
      default: return 'bg-green-600';
    }
  };

  const getAlertBorder = (level) => {
    switch(level) {
      case 'CRITICAL': return 'border-red-600';
      case 'WARNING': return 'border-yellow-600';
      default: return 'border-green-600';
    }
  };

  const getAlertIcon = (level) => {
    switch(level) {
      case 'CRITICAL': return '🚨';
      case 'WARNING': return '⚠️';
      default: return '✅';
    }
  };

  const getAlertText = (level) => {
    switch(level) {
      case 'CRITICAL': return 'Critical Risk - Immediate Action Required';
      case 'WARNING': return 'Moderate Risk - Monitor Closely';
      default: return 'Low Risk - Continue Routine Care';
    }
  };

  const getClinicalRecommendation = (level) => {
    if (level === 'CRITICAL') {
      return '🚨 EMERGENCY: Immediate ICU consultation required';
    } else if (level === 'WARNING') {
      return '⚠️ Increase monitoring frequency and consult clinical team';
    } else {
      return '✅ Continue standard care and routine monitoring';
    }
  };

  // Feature labels for display
  const featureLabels = {
    heart_rate: 'Heart Rate (bpm)',
    sbp: 'SBP (mmHg)',
    dbp: 'DBP (mmHg)',
    gcs: 'GCS Score',
    lactate: 'Lactate (mmol/L)',
    urine_output: 'Urine Output (mL)',
    fio2: 'FiO₂ (%)',
    creatinine: 'Creatinine (mg/dL)'
  };

  const featurePlaceholders = {
    heart_rate: '75',
    sbp: '120',
    dbp: '80',
    gcs: '13',
    lactate: '1.5',
    urine_output: '40',
    fio2: '35',
    creatinine: '1.0'
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  const hasPrediction = prediction !== null;

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">🏥 ICU Patient Assessment</h1>
            <p className="text-sm text-gray-500">Enter vitals to predict deterioration risk</p>
          </div>
          <div className="text-right text-sm">
            <p className="text-gray-500">{new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}</p>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white p-4 rounded-xl shadow">
            <p className="text-gray-500 text-xs">Total Predictions</p>
            <p className="text-2xl font-bold text-blue-600">{stats?.total_predictions || 0}</p>
          </div>
          <div className="bg-white p-4 rounded-xl shadow">
            <p className="text-gray-500 text-xs">High Risk (24h)</p>
            <p className="text-2xl font-bold text-red-600">{stats?.high_risk_patients || 0}</p>
          </div>
          <div className="bg-white p-4 rounded-xl shadow">
            <p className="text-gray-500 text-xs">Critical Alerts</p>
            <p className="text-2xl font-bold text-yellow-600">{stats?.critical_alerts || 0}</p>
          </div>
          <div className="bg-white p-4 rounded-xl shadow">
            <p className="text-gray-500 text-xs">Avg Risk Score</p>
            <p className="text-2xl font-bold text-blue-600">{((stats?.avg_risk_score || 0) * 100).toFixed(1)}%</p>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Patient Data Entry */}
          <div className="lg:col-span-2">
            <div className="bg-white p-6 rounded-xl shadow">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold">Enter Patient Vitals</h2>
                <button
                  onClick={resetForm}
                  className="text-sm text-gray-500 hover:text-gray-700 transition"
                >
                  🔄 Reset
                </button>
              </div>

              <form onSubmit={handlePredict}>
                {/* Patient ID and Name */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  <div>
                    <p className="text-xs text-gray-500">Patient ID (Auto)</p>
                    <p className="font-mono text-sm font-medium text-blue-600">{patientId}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Patient Name <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      name="patient_name"
                      value={patientData.patient_name}
                      onChange={handleChange}
                      placeholder="Enter patient name"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm"
                      required
                    />
                  </div>
                </div>

                {/* Vitals Input */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.keys(featureLabels).map((key) => (
                    <div key={key}>
                      <label className="block text-xs font-medium text-gray-700">
                        {featureLabels[key]}
                        <span className="text-red-500 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        name={key}
                        value={patientData[key]}
                        onChange={handleChange}
                        placeholder={featurePlaceholders[key]}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm"
                        step="0.1"
                        required
                      />
                    </div>
                  ))}
                </div>

                <button
                  type="submit"
                  disabled={predicting}
                  className="w-full mt-4 bg-blue-600 text-white py-2 rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50"
                >
                  {predicting ? 'Analyzing...' : '🔍 Predict Risk'}
                </button>
              </form>
            </div>
          </div>

          {/* Right Column - Model Info */}
          <div className="lg:col-span-1">
            <div className="bg-white p-6 rounded-xl shadow">
              <h3 className="font-semibold text-gray-700 mb-3">📊 Model Info</h3>
              <div className="space-y-2 text-sm">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-blue-800 font-medium">CNN-LSTM + Attention</p>
                  <p className="text-blue-600 text-xs">AUC-ROC: 0.6945</p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-green-800 font-medium">Accuracy: 86.68%</p>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <p className="text-purple-800 font-medium">MIMIC-IV v3.1</p>
                </div>
                <div className="p-3 bg-yellow-50 rounded-lg">
                  <p className="text-yellow-800 font-medium">Best: Heart Rate (H6)</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Result */}
        {hasPrediction && (
          <div className={`mt-6 bg-white p-6 rounded-xl shadow border-l-8 ${getAlertBorder(prediction.alert_level)}`}>
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{getAlertIcon(prediction.alert_level)}</span>
                  <h3 className="text-xl font-bold">{patientData.patient_name}</h3>
                </div>
                <p className="text-gray-500 text-sm">ID: {prediction.patient_id}</p>
                <p className="text-gray-400 text-xs">{new Date(prediction.predicted_at).toLocaleString()}</p>
              </div>
              <div className={`px-4 py-2 rounded-full text-white font-bold ${getAlertColor(prediction.alert_level)}`}>
                {prediction.alert_level}
              </div>
            </div>

            {/* Vitals Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4 p-3 bg-gray-50 rounded-lg">
              {Object.entries(featureLabels).map(([key, label]) => (
                <div key={key}>
                  <p className="text-xs text-gray-500">{label}</p>
                  <p className="font-semibold">{patientData[key] || '--'}</p>
                </div>
              ))}
            </div>

            {/* Risk Score */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
              <div className="bg-gray-50 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-500">Risk Score</p>
                <p className="text-2xl font-bold">{prediction.risk_score.toFixed(3)}</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-500">Risk %</p>
                <p className="text-2xl font-bold">{prediction.risk_percentage.toFixed(1)}%</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-500">Confidence</p>
                <p className="text-2xl font-bold">{prediction.confidence}</p>
              </div>
            </div>

            {/* Risk Gauge */}
            <div className="mt-4">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>{getAlertText(prediction.alert_level)}</span>
                <span>{prediction.risk_percentage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${
                    prediction.risk_score > 0.7 ? 'bg-red-600' :
                    prediction.risk_score > 0.5 ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}
                  style={{ width: `${prediction.risk_percentage}%` }}
                ></div>
              </div>
            </div>

            {/* Recommendation */}
            <div className={`mt-4 p-3 rounded-lg border-l-4 ${
              prediction.alert_level === 'CRITICAL' ? 'border-red-500 bg-red-50' :
              prediction.alert_level === 'WARNING' ? 'border-yellow-500 bg-yellow-50' :
              'border-green-500 bg-green-50'
            }`}>
              <p className={`text-sm font-medium ${
                prediction.alert_level === 'CRITICAL' ? 'text-red-800' :
                prediction.alert_level === 'WARNING' ? 'text-yellow-800' :
                'text-green-800'
              }`}>
                {getClinicalRecommendation(prediction.alert_level)}
              </p>
            </div>
          </div>
        )}

        {!hasPrediction && (
          <div className="mt-6 bg-white p-6 rounded-xl shadow text-center text-gray-500 py-8">
            <div className="text-4xl mb-3">📋</div>
            <h3 className="text-lg font-semibold">Enter Patient Vitals</h3>
            <p className="text-sm">Fill in the vitals and click "Predict Risk"</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
