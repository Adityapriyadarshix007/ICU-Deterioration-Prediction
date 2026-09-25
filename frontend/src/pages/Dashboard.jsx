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
import AssessmentForm, { StayIdAssessmentForm } from './Dashboard/components/AssessmentForm';
import PredictionCard from './Dashboard/components/PredictionCard';
import RecentPredictions from './Dashboard/components/RecentPredictions';
import ModelInfoCard from './Dashboard/components/ModelInfoCard';
import SystemStatus from './Dashboard/components/SystemStatus';

// Derive a confidence score from the model probability.
// Low confidence near 0.5 (borderline), high confidence near 0 or 1 (clear-cut).
const deriveConfidence = (probability) => {
  const p = Number(probability);
  if (!Number.isFinite(p)) return 0.5;
  return Math.min(1, 0.5 + Math.abs(p - 0.5));
};

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
    heart_rate: '', respiratory_rate: '', spo2: '', temperature: '',
    sbp: '', dbp: '', map: '',
    fio2: '', pao2: '',
    gcs_eyes: '', gcs_verbal: '', gcs_motor: '', gcs: '',
    creatinine: '', bun: '', sodium: '', potassium: '', chloride: '',
    bicarbonate: '', glucose: '',
    hemoglobin: '', hematocrit: '', wbc: '', platelets: '',
    inr: '', ptt: '',
    bilirubin: '', alt: '', ast: '',
    lactate: '', urine_output: '',
    norepinephrine_max_rate: '', epinephrine_max_rate: '',
    dopamine_max_rate: '', dobutamine_max_rate: '',
    vasopressin_max_rate: '', phenylephrine_max_rate: '',
  });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictingSteps, setPredictingSteps] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [systemStatus] = useState({
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
      // Keep any previously loaded stats on failure
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
      // Keep prior list on failure
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

  const handleRefresh = async () => {
    setRefreshing(true);
    setPrediction(null);
    setPredictingSteps([]);

    // Blank ALL vitals keys without removing any (preserves the 33-field schema)
    setVitals((prev) => {
      const blanked = {};
      Object.keys(prev).forEach((k) => { blanked[k] = ''; });
      return blanked;
    });

    // Clear demographic text inputs (keep MRN / patientId as-is)
    setPatientName('');
    setPatientAge('');
    setPatientGender('');
    setPatientDiagnosis('');
    setPatientRoom('');

    // Reload stats + recent in parallel; the loaders themselves preserve data on failure
    await Promise.allSettled([loadStats(), loadRecentPredictions()]);

    setRefreshing(false);
    toast.success('Dashboard refreshed');
  };

  const handleVitalChange = useCallback((e) => {
    const { name, value } = e.target;
    setVitals(prev => ({ ...prev, [name]: value }));
  }, []);

  const simulatePredictionSteps = async () => {
    const steps = [
      'Collecting vital signs...',
      'Running LightGBM model...',
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

    if (!patientName) { toast.error('Please enter patient name'); return; }
    if (!patientAge) { toast.error('Please enter patient age'); return; }
    if (!patientGender) { toast.error('Please select patient gender'); return; }
    if (!patientRoom) { toast.error('Please enter patient room'); return; }
    if (!patientDiagnosis) { toast.error('Please enter patient diagnosis'); return; }

    const validationErrors = validateVitals(vitals);
    if (validationErrors.length > 0) {
      validationErrors.forEach(err => toast.error(err));
      return;
    }

    setPredicting(true);
    setPrediction(null);
    await simulatePredictionSteps();

    try {
      // Helper: numeric -> number; blank -> omit (backend treats as missing)
      const num = (v) => {
        if (v === '' || v === null || v === undefined) return undefined;
        const f = parseFloat(v);
        return Number.isFinite(f) ? f : undefined;
      };

      const data = {
        patient_id: patientId,
        patient_name: patientName,
        age: parseInt(patientAge),
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        // vitals
        heart_rate: num(vitals.heart_rate),
        respiratory_rate: num(vitals.respiratory_rate),
        spo2: num(vitals.spo2),
        temperature: num(vitals.temperature),
        sbp: num(vitals.sbp),
        dbp: num(vitals.dbp),
        map: num(vitals.map),
        // respiratory
        fio2: num(vitals.fio2),
        pao2: num(vitals.pao2),
        // neuro
        gcs_eyes: num(vitals.gcs_eyes),
        gcs_verbal: num(vitals.gcs_verbal),
        gcs_motor: num(vitals.gcs_motor),
        gcs: num(vitals.gcs),
        // chemistry
        creatinine: num(vitals.creatinine),
        bun: num(vitals.bun),
        sodium: num(vitals.sodium),
        potassium: num(vitals.potassium),
        chloride: num(vitals.chloride),
        bicarbonate: num(vitals.bicarbonate),
        glucose: num(vitals.glucose),
        // hematology
        hemoglobin: num(vitals.hemoglobin),
        hematocrit: num(vitals.hematocrit),
        wbc: num(vitals.wbc),
        platelets: num(vitals.platelets),
        // coagulation
        inr: num(vitals.inr),
        ptt: num(vitals.ptt),
        // liver
        bilirubin: num(vitals.bilirubin),
        alt: num(vitals.alt),
        ast: num(vitals.ast),
        // misc
        lactate: num(vitals.lactate),
        urine_output: num(vitals.urine_output),
        // vasopressors
        norepinephrine_max_rate: num(vitals.norepinephrine_max_rate),
        epinephrine_max_rate: num(vitals.epinephrine_max_rate),
        dopamine_max_rate: num(vitals.dopamine_max_rate),
        dobutamine_max_rate: num(vitals.dobutamine_max_rate),
        vasopressin_max_rate: num(vitals.vasopressin_max_rate),
        phenylephrine_max_rate: num(vitals.phenylephrine_max_rate),
      };

      // Remove undefined keys so the JSON body is clean
      Object.keys(data).forEach((k) => {
        if (data[k] === undefined) delete data[k];
      });

      console.log('📤 Sending prediction data:', data);

      const response = await api.predict(data);
      const predictionData = response.data || {};

      console.log('📥 Received prediction:', predictionData);

      const probability = predictionData.risk_score ?? predictionData.probability ?? 0.5;

      setPrediction({
        patient_id: patientId,
        patient_name: patientName,
        age: patientAge,
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        risk_score: probability,
        risk_percentage: probability * 100,
        alert_level: predictionData.alert_level || predictionData.risk_band || 'LOW',
        confidence: deriveConfidence(probability),
        features: predictionData.features || data,
        shap_values: predictionData.shap_values || null,
        prediction_time: new Date().toISOString()
      });

      toast.success('Clinical risk assessment generated successfully');
      loadRecentPredictions();
    } catch (error) {
      console.error('❌ Prediction failed:', error);

      // Extract a string from the API error, handling Pydantic's
      // 422 array-of-objects shape: [{loc, msg, type, ...}, ...]
      const raw = error.response?.data?.detail;
      let errorMsg;
      if (Array.isArray(raw)) {
        // Pydantic validation error
        errorMsg = raw
          .map((e) => {
            const field = Array.isArray(e?.loc) ? e.loc.slice(-1)[0] : 'field';
            return `${field}: ${e?.msg || 'invalid'}`;
          })
          .join('; ');
      } else if (typeof raw === 'string') {
        errorMsg = raw;
      } else {
        errorMsg = error.response?.data?.message ||
                   error.message ||
                   'Unable to generate prediction. Please verify patient information and try again.';
      }

      toast.error(errorMsg);

      // Fallback placeholder — clearly marked so it is not mistaken for model output
      setPrediction({
        patient_id: patientId,
        patient_name: patientName,
        age: patientAge,
        gender: patientGender,
        room: patientRoom,
        diagnosis: patientDiagnosis,
        risk_score: 0.35,
        risk_percentage: 35,
        alert_level: 'LOW',
        confidence: 0.5,
        is_fallback: true,
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

  // --- additive: stay_id flow ---
  const normalizeStayIdResult = (raw) => {
    const probability = Number(raw?.probability ?? 0);
    const riskBand = raw?.risk_band || 'LOW';
    const topFeatures = Array.isArray(raw?.top_features) ? raw.top_features : [];
    const shapObject = topFeatures.reduce((acc, f) => {
      if (f && typeof f.feature === 'string') {
        acc[f.feature] = Number(f.contribution ?? 0);
      }
      return acc;
    }, {});
    return {
      patient_id: String(raw?.stay_id ?? ''),
      patient_name: `Stay ${raw?.stay_id ?? ''}`,
      risk_score: probability,
      risk_percentage: probability * 100,
      alert_level: riskBand,
      confidence: deriveConfidence(probability),
      features: {},
      shap_values: shapObject,
      threshold: raw?.threshold ?? 0.5,
      stay_id: raw?.stay_id,
      prediction_time: new Date().toISOString(),
    };
  };

  const handleStayIdPredict = async (rawResponse) => {
    try {
      setPrediction(normalizeStayIdResult(rawResponse));
      toast.success('Prediction generated from ICU stay ID');
      loadRecentPredictions();
    } catch (err) {
      console.error('Stay-ID prediction failed:', err);
      toast.error('Could not process stay-ID prediction');
    }
  };

  // --- additive: load features from chart into the manual form ---
  const handleLoadFromChart = (features) => {
    // Map backend feature names -> form field names where they differ.
    const FEATURE_TO_FORM = {
      gcs_total: 'gcs',
    };

    // Rounding precision per form field (matches the form's UX).
    const DECIMALS = {
      heart_rate: 0, respiratory_rate: 0, spo2: 0, temperature: 1,
      sbp: 0, dbp: 0, map: 0,
      fio2: 0, pao2: 0,
      gcs_eyes: 0, gcs_verbal: 0, gcs_motor: 0, gcs: 0,
      creatinine: 2, bun: 0, sodium: 0, potassium: 1, chloride: 0,
      bicarbonate: 0, glucose: 0,
      hemoglobin: 1, hematocrit: 1, wbc: 1, platelets: 0,
      inr: 1, ptt: 0,
      bilirubin: 1, alt: 0, ast: 0,
      lactate: 1, urine_output: 0,
      norepinephrine_max_rate: 2, epinephrine_max_rate: 2,
      dopamine_max_rate: 1, dobutamine_max_rate: 1,
      vasopressin_max_rate: 3, phenylephrine_max_rate: 2,
    };

    const roundTo = (value, decimals) => {
      const factor = Math.pow(10, decimals);
      return Math.round(value * factor) / factor;
    };

    let populated = 0;

    setVitals((prev) => {
      const merged = { ...prev };
      Object.keys(features).forEach((backendKey) => {
        const formKey = FEATURE_TO_FORM[backendKey] || backendKey;
        if (!(formKey in merged)) return;

        const raw = features[backendKey];
        if (raw == null || Number.isNaN(raw)) return;

        const decimals = DECIMALS[formKey] ?? 2;
        merged[formKey] = String(roundTo(Number(raw), decimals));
        populated += 1;
      });
      return merged;
    });

    // Auto-fill top-level demographics (not part of vitals state)
    if ('age' in features && features.age != null) {
      setPatientAge(String(Math.round(features.age)));
    }
    if ('gender_male' in features && features.gender_male != null) {
      setPatientGender(features.gender_male >= 0.5 ? 'Male' : 'Female');
    }

    if (populated === 0) {
      toast.error('No matching fields to load — check that the chart has data');
    } else {
      toast.success(`Loaded ${populated} values from ICU chart — review before predicting`);
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
                <span className="text-sm text-gray-500">Model v3.0</span>
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

            {/* additive: one-click prediction by ICU stay ID */}
            <div className="mt-8">
              <StayIdAssessmentForm
                onResult={handleStayIdPredict}
                onLoadFromChart={handleLoadFromChart}
              />
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
