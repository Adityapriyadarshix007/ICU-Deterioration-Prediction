import React from 'react';
import { motion } from 'framer-motion';
import CountUp from 'react-countup';

const RiskGauge = ({ value, maxValue = 100, size = 140, label = 'Risk Score' }) => {
  const percentage = Math.max(0, Math.min(100, (value / maxValue) * 100));
  const circumference = 2 * Math.PI * 45;
  const offset = circumference - (percentage / 100) * circumference;
  
  const getColor = (val) => {
    if (val > 70) return '#ef4444';
    if (val > 40) return '#f59e0b';
    return '#22c55e';
  };

  const getRiskLevel = (val) => {
    if (val > 70) return 'HIGH';
    if (val > 40) return 'MEDIUM';
    return 'LOW';
  };

  const getRiskText = (val) => {
    if (val > 70) return 'Critical Risk';
    if (val > 40) return 'Elevated Risk';
    return 'Low Risk';
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative inline-flex items-center justify-center">
        <svg width={size} height={size} viewBox="0 0 120 120" className="transform -rotate-90">
          <circle cx="60" cy="60" r="45" fill="none" stroke="#e5e7eb" strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r="45"
            fill="none"
            stroke={getColor(percentage)}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span 
            className="text-2xl font-bold text-gray-900"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <CountUp end={Math.round(percentage)} duration={1.5} suffix="%" />
          </motion.span>
          <span className="text-xs text-gray-500">{label}</span>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs font-medium text-green-600">LOW</span>
        <div className="w-32 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div className="h-full w-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full" />
        </div>
        <span className="text-xs font-medium text-red-600">HIGH</span>
      </div>
      <div className="mt-2 flex flex-col items-center">
        <span className={`text-sm font-bold ${getRiskLevel(percentage) === 'HIGH' ? 'text-red-600' : getRiskLevel(percentage) === 'MEDIUM' ? 'text-yellow-600' : 'text-green-600'}`}>
          {getRiskText(percentage)}
        </span>
        <span className="text-xs text-gray-400">
          {getRiskLevel(percentage)} RISK
        </span>
      </div>
    </div>
  );
};

export default RiskGauge;
