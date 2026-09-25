export const CLINICAL_RANGES = {
  // ============================================================
  // Vitals
  // ============================================================
  heart_rate: { 
    min: 40, max: 160, unit: 'bpm', label: 'Heart Rate', 
    integer: true, minAllowed: 20, maxAllowed: 200 
  },
  respiratory_rate: { 
    min: 8, max: 40, unit: 'breaths/min', label: 'Respiratory Rate', 
    integer: true, minAllowed: 4, maxAllowed: 60 
  },
  spo2: { 
    min: 70, max: 100, unit: '%', label: 'SpO₂', 
    integer: true, minAllowed: 50, maxAllowed: 100 
  },
  temperature: { 
    min: 95, max: 104, unit: '°F', label: 'Temperature', 
    integer: false, minAllowed: 86, maxAllowed: 110 
  },
  sbp: { 
    min: 70, max: 220, unit: 'mmHg', label: 'Systolic BP', 
    integer: true, minAllowed: 50, maxAllowed: 260 
  },
  dbp: { 
    min: 40, max: 130, unit: 'mmHg', label: 'Diastolic BP', 
    integer: true, minAllowed: 30, maxAllowed: 160 
  },
  map: { 
    min: 40, max: 140, unit: 'mmHg', label: 'MAP', 
    integer: true, minAllowed: 20, maxAllowed: 180 
  },

  // ============================================================
  // Respiratory
  // ============================================================
  fio2: { 
    min: 21, max: 100, unit: '%', label: 'FiO₂', 
    integer: true, minAllowed: 15, maxAllowed: 100 
  },
  pao2: { 
    min: 40, max: 500, unit: 'mmHg', label: 'PaO₂', 
    integer: true, minAllowed: 20, maxAllowed: 600 
  },

  // ============================================================
  // Neuro (components + total; components override total if given)
  // ============================================================
  gcs_eyes: { 
    min: 1, max: 4, unit: '', label: 'GCS – Eyes', 
    integer: true, minAllowed: 1, maxAllowed: 4 
  },
  gcs_verbal: { 
    min: 1, max: 5, unit: '', label: 'GCS – Verbal', 
    integer: true, minAllowed: 1, maxAllowed: 5 
  },
  gcs_motor: { 
    min: 1, max: 6, unit: '', label: 'GCS – Motor', 
    integer: true, minAllowed: 1, maxAllowed: 6 
  },
  gcs: { 
    min: 3, max: 15, unit: '', label: 'GCS Total', 
    integer: true, minAllowed: 3, maxAllowed: 15 
  },

  // ============================================================
  // Chemistry
  // ============================================================
  creatinine: { 
    min: 0.3, max: 10, unit: 'mg/dL', label: 'Creatinine', 
    integer: false, minAllowed: 0.1, maxAllowed: 15 
  },
  bun: { 
    min: 5, max: 100, unit: 'mg/dL', label: 'BUN', 
    integer: true, minAllowed: 1, maxAllowed: 200 
  },
  sodium: { 
    min: 120, max: 160, unit: 'mEq/L', label: 'Sodium', 
    integer: true, minAllowed: 90, maxAllowed: 200 
  },
  potassium: { 
    min: 2.5, max: 6.5, unit: 'mEq/L', label: 'Potassium', 
    integer: false, minAllowed: 1, maxAllowed: 12 
  },
  chloride: { 
    min: 90, max: 115, unit: 'mEq/L', label: 'Chloride', 
    integer: true, minAllowed: 50, maxAllowed: 200 
  },
  bicarbonate: { 
    min: 10, max: 35, unit: 'mEq/L', label: 'Bicarbonate', 
    integer: true, minAllowed: 0, maxAllowed: 60 
  },
  glucose: { 
    min: 50, max: 500, unit: 'mg/dL', label: 'Glucose', 
    integer: true, minAllowed: 0, maxAllowed: 1500 
  },

  // ============================================================
  // Hematology
  // ============================================================
  hemoglobin: { 
    min: 5, max: 20, unit: 'g/dL', label: 'Hemoglobin', 
    integer: false, minAllowed: 0, maxAllowed: 25 
  },
  hematocrit: { 
    min: 15, max: 60, unit: '%', label: 'Hematocrit', 
    integer: false, minAllowed: 0, maxAllowed: 70 
  },
  wbc: { 
    min: 0.5, max: 40, unit: 'K/µL', label: 'WBC', 
    integer: false, minAllowed: 0, maxAllowed: 200 
  },
  platelets: { 
    min: 10, max: 600, unit: 'K/µL', label: 'Platelets', 
    integer: true, minAllowed: 0, maxAllowed: 1500 
  },

  // ============================================================
  // Coagulation
  // ============================================================
  inr: { 
    min: 0.8, max: 5, unit: '', label: 'INR', 
    integer: false, minAllowed: 0, maxAllowed: 20 
  },
  ptt: { 
    min: 20, max: 100, unit: 'sec', label: 'PTT', 
    integer: true, minAllowed: 0, maxAllowed: 200 
  },

  // ============================================================
  // Liver
  // ============================================================
  bilirubin: { 
    min: 0.1, max: 20, unit: 'mg/dL', label: 'Bilirubin', 
    integer: false, minAllowed: 0, maxAllowed: 60 
  },
  alt: { 
    min: 5, max: 500, unit: 'U/L', label: 'ALT', 
    integer: true, minAllowed: 0, maxAllowed: 5000 
  },
  ast: { 
    min: 5, max: 500, unit: 'U/L', label: 'AST', 
    integer: true, minAllowed: 0, maxAllowed: 5000 
  },

  // ============================================================
  // Misc
  // ============================================================
  lactate: { 
    min: 0.5, max: 20, unit: 'mmol/L', label: 'Lactate', 
    integer: false, minAllowed: 0.1, maxAllowed: 25 
  },
  urine_output: { 
    min: 10, max: 200, unit: 'mL', label: 'Urine Output', 
    integer: true, minAllowed: 5, maxAllowed: 250 
  },

  // ============================================================
  // Vasopressors
  // ============================================================
  norepinephrine_max_rate: { 
    min: 0.0, max: 0.5, unit: 'mcg/kg/min', label: 'Norepinephrine (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 5 
  },
  epinephrine_max_rate: { 
    min: 0.0, max: 0.5, unit: 'mcg/kg/min', label: 'Epinephrine (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 5 
  },
  dopamine_max_rate: { 
    min: 0.0, max: 20, unit: 'mcg/kg/min', label: 'Dopamine (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 50 
  },
  dobutamine_max_rate: { 
    min: 0.0, max: 20, unit: 'mcg/kg/min', label: 'Dobutamine (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 50 
  },
  vasopressin_max_rate: { 
    min: 0.0, max: 0.1, unit: 'units/min', label: 'Vasopressin (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 5 
  },
  phenylephrine_max_rate: { 
    min: 0.0, max: 5, unit: 'mcg/kg/min', label: 'Phenylephrine (max rate)', 
    integer: false, minAllowed: 0, maxAllowed: 5 
  },
};

export const MODEL_INFO = {
  name: 'LightGBM',
  version: 'v3.0',
  aucRoc: 0.8000,
  auprc: 0.5701,
  threshold: 0.5,
  sensitivity: 67.39,
  specificity: 75.42,
  dataset: 'MIMIC-IV v3.1',
  samples: 54544,
  features: 160,
  predictionWindow: '6-18 hours (ΔSOFA ≥ 2)',
  trainingDate: 'September 2026',
  calibration: 'Isotonic Regression',
  prevalence: 0.1891,
};
