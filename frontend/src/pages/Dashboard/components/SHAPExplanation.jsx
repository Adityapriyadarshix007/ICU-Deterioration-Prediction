import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

const SHAP_FEATURES = {
  lactate: { label: 'Lactate', color: '#ef4444', icon: TrendingUp },
  gcs: { label: 'GCS', color: '#f59e0b', icon: TrendingDown },
  creatinine: { label: 'Creatinine', color: '#3b82f6', icon: TrendingUp },
  heart_rate: { label: 'Heart Rate', color: '#8b5cf6', icon: TrendingUp },
  sbp: { label: 'Systolic BP', color: '#ec4899', icon: TrendingDown },
  fio2: { label: 'FiO₂', color: '#14b8a6', icon: TrendingUp },
  urine_output: { label: 'Urine Output', color: '#f97316', icon: TrendingDown },
};

const SHAPExplanation = ({ shapValues }) => {
  if (!shapValues || Object.keys(shapValues).length === 0) {
    return <p className="text-sm text-gray-400">No SHAP values available</p>;
  }

  const sorted = Object.entries(shapValues)
    .filter(([_, value]) => Math.abs(value) > 0.02)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5);

  if (sorted.length === 0) {
    return <p className="text-sm text-gray-400">No significant factors identified</p>;
  }

  return (
    <div className="space-y-3">
      {sorted.map(([key, value], index) => {
        const feature = SHAP_FEATURES[key];
        if (!feature) return null;
        const Icon = feature.icon;
        const isPositive = value > 0;
        const absValue = Math.abs(value);
        const percent = Math.min(Math.round(absValue * 100), 100);

        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
          >
            <div className="flex items-center gap-3 flex-1">
              <div className="p-1.5 rounded-lg" style={{ background: `${feature.color}20` }}>
                <Icon className="w-4 h-4" style={{ color: feature.color }} />
              </div>
              <span className="text-sm font-medium text-gray-700">{feature.label}</span>
            </div>
            <div className="flex items-center gap-3 flex-1">
              <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percent}%` }}
                  className={`h-full rounded-full ${isPositive ? 'bg-red-500' : 'bg-green-500'}`}
                  transition={{ duration: 0.8, delay: index * 0.05 }}
                />
              </div>
              <span className={`text-sm font-semibold w-14 text-right ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                {isPositive ? '↑' : '↓'} {percent}%
              </span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

export default SHAPExplanation;
