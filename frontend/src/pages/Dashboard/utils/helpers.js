import { CLINICAL_RANGES } from './constants';

export const formatPatientId = () => {
  const date = new Date();
  const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
  const random = String(Math.floor(Math.random() * 1000)).padStart(3, '0');
  return `PAT-${dateStr}-${random}`;
};

export const formatName = (name) => {
  if (!name) return 'User';
  return name.split(' ').map(n => n.charAt(0).toUpperCase() + n.slice(1).toLowerCase()).join(' ');
};

export const getRiskColor = (level) => {
  const colors = {
    CRITICAL: 'critical',
    WARNING: 'warning',
    STABLE: 'stable'
  };
  return colors[level] || 'stable';
};

export const getRiskScoreColor = (score) => {
  if (score > 70) return 'text-red-600';
  if (score > 40) return 'text-yellow-600';
  return 'text-green-600';
};

// Improved recommendation based on score only (cleaner logic)
export const getRecommendation = (score) => {
  const percentage = typeof score === 'number' ? score : (score || 0);
  
  if (percentage >= 70) {
    return {
      title: '🚨 Immediate ICU Review Required',
      actions: [
        'Notify ICU attending physician immediately',
        'Repeat vital signs within 15 minutes',
        'Monitor lactate trend closely',
        'Consider vasopressor assessment',
        'Prepare for potential ICU transfer'
      ]
    };
  } else if (percentage >= 40) {
    return {
      title: '⚠️ Enhanced Monitoring Required',
      actions: [
        'Increase vital sign monitoring to every 30 minutes',
        'Monitor GCS and lactate trends',
        'Check creatinine and urine output in 2 hours',
        'Review with senior clinician',
        'Document clinical concerns'
      ]
    };
  } else {
    return {
      title: '✅ Continue Routine Care',
      actions: [
        'Continue standard monitoring protocol',
        'Reassess in 4 hours',
        'Maintain current treatment plan',
        'Document stable condition',
        'Patient education as needed'
      ]
    };
  }
};

export const getAlertLevel = (score) => {
  const percentage = typeof score === 'number' ? score : (score || 0);
  if (percentage >= 70) return 'CRITICAL';
  if (percentage >= 40) return 'WARNING';
  return 'STABLE';
};

export const validateVitals = (vitals) => {
  const errors = [];
  
  Object.entries(vitals).forEach(([key, value]) => {
    const range = CLINICAL_RANGES[key];
    if (!range) return;
    
    if (value === '' || value === null || value === undefined) {
      errors.push(`${range.label} is required`);
      return;
    }
    
    const num = parseFloat(value);
    if (isNaN(num)) {
      errors.push(`${range.label} must be a valid number`);
      return;
    }
    
    if (key === 'gcs') {
      if (num < 3 || num > 15) {
        errors.push(`GCS must be between 3 and 15 (got ${num})`);
      }
      if (!Number.isInteger(num)) {
        errors.push(`GCS must be a whole number (got ${num})`);
      }
      return;
    }
    
    if (range.integer && !Number.isInteger(num)) {
      errors.push(`${range.label} must be a whole number (got ${num})`);
    }
    
    if (num < range.minAllowed || num > range.maxAllowed) {
      errors.push(`${range.label} should be between ${range.min} and ${range.max} ${range.unit} (got ${num})`);
    }
  });
  
  return errors;
};
