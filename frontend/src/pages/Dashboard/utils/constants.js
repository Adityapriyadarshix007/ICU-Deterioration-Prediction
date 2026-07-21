export const CLINICAL_RANGES = {
  heart_rate: { 
    min: 40, max: 160, unit: 'bpm', label: 'Heart Rate', 
    integer: true, minAllowed: 20, maxAllowed: 200 
  },
  sbp: { 
    min: 70, max: 220, unit: 'mmHg', label: 'Systolic BP', 
    integer: true, minAllowed: 50, maxAllowed: 260 
  },
  dbp: { 
    min: 40, max: 130, unit: 'mmHg', label: 'Diastolic BP', 
    integer: true, minAllowed: 30, maxAllowed: 160 
  },
  gcs: { 
    min: 3, max: 15, unit: '', label: 'Glasgow Coma Scale', 
    integer: true, minAllowed: 3, maxAllowed: 15 
  },
  lactate: { 
    min: 0.5, max: 20, unit: 'mmol/L', label: 'Lactate', 
    integer: false, minAllowed: 0.1, maxAllowed: 25 
  },
  creatinine: { 
    min: 0.3, max: 10, unit: 'mg/dL', label: 'Creatinine', 
    integer: false, minAllowed: 0.1, maxAllowed: 15 
  },
  fio2: { 
    min: 21, max: 100, unit: '%', label: 'FiO₂', 
    integer: true, minAllowed: 15, maxAllowed: 100 
  },
  urine_output: { 
    min: 10, max: 200, unit: 'mL', label: 'Urine Output', 
    integer: true, minAllowed: 5, maxAllowed: 250 
  },
};

export const MODEL_INFO = {
  name: 'CatBoost',
  version: 'v2.0',
  aucRoc: 0.7013,
  threshold: 0.459,
  sensitivity: 68.47,
  specificity: 62.03,
  dataset: 'MIMIC-IV v3.1',
  samples: 57515,
  features: 32,
  predictionWindow: '24 hours',
  trainingDate: 'July 2026',
  calibration: 'Isotonic Regression',
};
