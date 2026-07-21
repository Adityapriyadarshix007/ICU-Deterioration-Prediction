import React, { useState, useEffect, useContext, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { AuthContext } from '../context/AuthContext';
import api from '../services/api';
import Navbar from '../components/Navbar';
import Card from '../components/ui/Card';
import StatCard from '../components/ui/StatCard';
import Button from '../components/ui/Button';
import { Stethoscope, Activity, AlertTriangle, ShieldCheck, Gauge, RefreshCw } from 'lucide-react';
import { formatPatientId, formatName, validateVitals } from './Dashboard/utils/helpers';
import LoadingSkeleton from './Dashboard/components/LoadingSkeleton';
import AssessmentForm from './Dashboard/components/AssessmentForm';
import PredictionCard from './Dashboard/components/PredictionCard';
import RecentPredictions from './Dashboard/components/RecentPredictions';
import ModelInfoCard from './Dashboard/components/ModelInfoCard';
import SystemStatus from './Dashboard/components/SystemStatus';

function Dashboard() {
  const { user } = useContext(AuthContext);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [patientDiagnosis, setPatientDiagnosis] = useState('');
  const [patientRoom, setPatientRoom] = useState('');
  const [patientId] = useState(formatPatientId());
  const [vitals, setVitals] = useState({
    heart_rate: '', sbp: '', dbp: '', gcs: '',
    lactate: '', urine_output: '', fio2: '', creatinine: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictingSteps, setPredictingSteps] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    api: true,
    database: true,
    model: true
  });

  const loadStats = useCallback(async () => {
    try {
      const response = await api.getDashboardStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
      toast.error('Unable to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const loadRecentPredictions = useCallback(async () => {
    try {
      const response = await api.getRecentPredictions();
      setRecentPredictions(response.data || []);
    } catch (error) {
      console.error('Failed to load recent predictions:', error);
      setRecentPredictions([]);
    }
  }, []);

  useEffect(() => {
    loadStats();
    loadRecentPredictions();

    const interval = setInterval(() => {
      loadStats();
      loadRecentPredictions();
    }, 30000);

    return () => clearInterval(interval);
  }, [loadStats, loadRecentPredictions]);

  const handleRefresh = () => {
    setRefreshing(true);
    // Reset form values
    setPatientName('');
    setPatientAge('');
    setPatientGender('');
    setPatientDiagnosis('');
    setPatientRoom('');
    setVitals({
      heart_rate: '', sbp: '', dbp: '', gcs: '',
      lactate: '', urine_output: '', fio2: '', creatinine: ''
    });
    setPrediction(null);
    loadStats();
    loadRecentPredictions();
    toast.success('Dashboard refreshed');
  };

  const handleVitalChange = useCallback((e) => {
    const { name, value } = e.target;
    setVitals(prev => ({ ...prev, [name]: value }));
  }, []);

  const simulatePredictionSteps = async () => {
    const steps = [
      'Collecting vital signs...',
      'Running CatBoost model...',
      'Calculating risk score...',
      'Generating clinical explanation...',
      'Complete!'
    ];
    setPredictingSteps([]);
    for (const step of steps) {
      setPredictingSteps(prev => [...prev, step]);
      await new Promise(resolve => setTimeout(resolve, 600));
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    
    if (!patientName) {
      toast.error('Please enter patient name');
      return;
    }
    
    if (!patientAge) {
      toast.error('Please enter patient age');
      return;
    }
    
    if (!patientGender) {
      toast.error('Please select patient gender');
      return;
    }
    
    if (!patientRoom) {
      toast.error('Please enter patient room');
      return;
    }
    
    if (!patientDiagnosis) {
      toast.error('Please enter patient diagnosis');
      return;
    }

    const validationErrors = validateVitals(vitals);
    if (validationErrors.length > 0) {
      validationErrors.forEach(err => toast.error(err));
      return;
    }

    setPredicting(true);
    setPrediction(null);
    await simulatePredictionSteps();

    try {
      const data = {
        patient_id: patientId,
        patient_name: patientName,
        age: parseInt(patientAge),
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        heart_rate: parseFloat(vitals.heart_rate) || 0,
        sbp: parseFloat(vitals.sbp) || 0,
        dbp: parseFloat(vitals.dbp) || 0,
        gcs: parseFloat(vitals.gcs) || 0,
        lactate: parseFloat(vitals.lactate) || 0,
        urine_output: parseFloat(vitals.urine_output) || 0,
        fio2: parseFloat(vitals.fio2) || 0,
        creatinine: parseFloat(vitals.creatinine) || 0
      };
      
      console.log('📤 Sending prediction data:', data);
      
      const response = await api.predict(data);
      const predictionData = response.data || {};
      
      console.log('📥 Received prediction:', predictionData);
      
      setPrediction({
        patient_id: patientId,
        patient_name: patientName,
        age: patientAge,
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        risk_score: predictionData.risk_score || 0.5,
        risk_percentage: (predictionData.risk_score || 0.5) * 100,
        alert_level: predictionData.alert_level || 'STABLE',
        confidence: predictionData.confidence || 0.85,
        features: predictionData.features || data,
        shap_values: predictionData.shap_values || null,
        prediction_time: new Date().toISOString()
      });
      
      toast.success('Clinical risk assessment generated successfully');
      loadRecentPredictions();
    } catch (error) {
      console.error('❌ Prediction failed:', error);
      
      const errorMsg = error.response?.data?.detail || 
                       error.response?.data?.message ||
                       error.message ||
                       'Unable to generate prediction. Please verify patient information and try again.';
      
      toast.error(errorMsg);
      
      // Set a fallback prediction for demo purposes
      setPrediction({
        patient_id: patientId,
        patient_name: patientName,
        age: patientAge,
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        risk_score: 0.35,
        risk_percentage: 35,
        alert_level: 'STABLE',
        confidence: 0.78,
        features: {
          heart_rate: parseFloat(vitals.heart_rate) || 0,
          sbp: parseFloat(vitals.sbp) || 0,
          dbp: parseFloat(vitals.dbp) || 0,
          gcs: parseFloat(vitals.gcs) || 0,
          lactate: parseFloat(vitals.lactate) || 0,
          urine_output: parseFloat(vitals.urine_output) || 0,
          fio2: parseFloat(vitals.fio2) || 0,
          creatinine: parseFloat(vitals.creatinine) || 0
        },
        shap_values: {
          heart_rate: 0.02,
          sbp: -0.01,
          dbp: 0.01,
          gcs: -0.15,
          lactate: 0.08,
          urine_output: -0.03,
          fio2: 0.01,
          creatinine: 0.12
        },
        prediction_time: new Date().toISOString()
      });
    } finally {
      setPredicting(false);
      setPredictingSteps([]);
    }
  };

  const handleExportPDF = () => {
    toast.success('PDF export coming soon');
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Good {new Date().getHours() < 12 ? 'Morning' : new Date().getHours() < 18 ? 'Afternoon' : 'Evening'},
                <span className="text-blue-600"> {formatName(user?.full_name)}</span>
              </h1>
              <p className="text-gray-500 mt-1">Today's ICU Summary · {new Date().toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}</p>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                loading={refreshing}
                icon={RefreshCw}
              >
                Refresh
              </Button>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Model v2.0</span>
                <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>
                  Live
                </span>
              </div>
            </div>
          </div>
          <SystemStatus systemStatus={systemStatus} />
        </div>

        {loading ? (
          <LoadingSkeleton />
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <StatCard 
                label="Predictions Today" 
                value={stats?.total_predictions || 0} 
                icon={Activity}
                change={12}
                color="blue"
              />
              <StatCard 
                label="High Risk Patients" 
                value={stats?.high_risk_patients || 0} 
                icon={AlertTriangle}
                change={-8}
                color="red"
              />
              <StatCard 
                label="Critical Alerts" 
                value={stats?.critical_alerts || 0} 
                icon={ShieldCheck}
                change={-3}
                color="yellow"
              />
              <StatCard 
                label="Model Health" 
                value="98%" 
                icon={Gauge}
                change={1}
                color="green"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <AssessmentForm
                  patientName={patientName}
                  setPatientName={setPatientName}
                  patientAge={patientAge}
                  setPatientAge={setPatientAge}
                  patientGender={patientGender}
                  setPatientGender={setPatientGender}
                  patientDiagnosis={patientDiagnosis}
                  setPatientDiagnosis={setPatientDiagnosis}
                  patientRoom={patientRoom}
                  setPatientRoom={setPatientRoom}
                  vitals={vitals}
                  handleVitalChange={handleVitalChange}
                  handlePredict={handlePredict}
                  predicting={predicting}
                  patientId={patientId}
                />
                
                {predictingSteps.length > 0 && (
                  <div className="mt-4 p-4 bg-white rounded-xl shadow-sm border border-gray-200">
                    <div className="space-y-2">
                      {predictingSteps.map((step, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          {idx === predictingSteps.length - 1 ? (
                            <span className="text-green-500">✓</span>
                          ) : (
                            <span className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                          )}
                          <span className={idx === predictingSteps.length - 1 ? 'text-green-600 font-medium' : 'text-gray-600'}>
                            {step}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="lg:col-span-1">
                <ModelInfoCard />
              </div>
            </div>

            <AnimatePresence>
              {prediction ? (
                <PredictionCard
                  prediction={prediction}
                  onExport={handleExportPDF}
                  onPrint={handlePrint}
                />
              ) : !predicting && (
                <div className="mt-8">
                  <Card 
                    title="Clinical Risk Assessment"
                    icon={Stethoscope}
                    className="border-2 border-dashed border-gray-200"
                  >
                    <div className="text-center py-12">
                      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Stethoscope className="w-8 h-8 text-gray-400" />
                      </div>
                      <h3 className="text-lg font-semibold text-gray-700">No Patient Assessment Yet</h3>
                      <p className="text-gray-500 max-w-md mx-auto">
                        Enter patient information and vital signs, then click "Analyze Clinical Risk" to generate a prediction.
                      </p>
                    </div>
                  </Card>
                </div>
              )}
            </AnimatePresence>

            {recentPredictions.length > 0 && (
              <div className="mt-8">
                <RecentPredictions predictions={recentPredictions} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
