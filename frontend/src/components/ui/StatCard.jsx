import React from 'react';

const colorMap = {
  blue: 'bg-blue-50 text-blue-600',
  red: 'bg-red-50 text-red-600',
  green: 'bg-green-50 text-green-600',
  yellow: 'bg-yellow-50 text-yellow-600',
  indigo: 'bg-indigo-50 text-indigo-600',
  purple: 'bg-purple-50 text-purple-600',
  pink: 'bg-pink-50 text-pink-600',
};

const StatCard = ({ 
  label, 
  value, 
  icon: Icon, 
  change,
  color = 'blue',
  className = ''
}) => {
  const isPositive = change > 0;
  const isNegative = change < 0;
  
  return (
    <div className={`bg-white rounded-xl shadow-sm p-6 border border-gray-100 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
        </div>
        {Icon && (
          <div className={`p-3 rounded-lg ${colorMap[color] || colorMap.blue}`}>
            <Icon className="w-6 h-6" />
          </div>
        )}
      </div>
      {change !== undefined && change !== 0 && (
        <div className="mt-3 flex items-center gap-1">
          <span className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? '↑' : '↓'} {Math.abs(change)}%
          </span>
          <span className="text-sm text-gray-400">from last week</span>
        </div>
      )}
    </div>
  );
};

export default StatCard;
