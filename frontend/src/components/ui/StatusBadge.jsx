import React from 'react';

const statusMap = {
  stable: 'bg-green-100 text-green-800',
  warning: 'bg-yellow-100 text-yellow-800',
  critical: 'bg-red-100 text-red-800',
  low: 'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-red-100 text-red-800',
};

const sizeMap = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-3 py-1 text-sm',
  lg: 'px-4 py-1.5 text-base',
};

const StatusBadge = ({ 
  status = 'stable', 
  label,
  size = 'md',
  className = '',
  ...props 
}) => {
  const statusClass = statusMap[status] || statusMap.stable;
  const sizeClass = sizeMap[size] || sizeMap.md;
  
  return (
    <span 
      className={`inline-flex items-center font-medium rounded-full ${statusClass} ${sizeClass} ${className}`}
      {...props}
    >
      {label || status.toUpperCase()}
    </span>
  );
};

export default StatusBadge;
