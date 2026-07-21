import React from 'react';
import { CheckCircle, XCircle } from 'lucide-react';

const SystemStatus = ({ systemStatus }) => {
  const defaultStatus = {
    api: true,
    database: true,
    model: true
  };

  const status = systemStatus || defaultStatus;

  return (
    <div className="mt-4 flex items-center gap-4 text-sm flex-wrap">
      <span className="text-gray-500 font-medium">System Status:</span>
      <div className="flex items-center gap-3 flex-wrap">
        <span className={`flex items-center gap-1 ${status.api ? 'text-green-600' : 'text-red-600'}`}>
          {status.api ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          API
        </span>
        <span className={`flex items-center gap-1 ${status.database ? 'text-green-600' : 'text-red-600'}`}>
          {status.database ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          Database
        </span>
        <span className={`flex items-center gap-1 ${status.model ? 'text-green-600' : 'text-red-600'}`}>
          {status.model ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          Prediction Engine
        </span>
      </div>
    </div>
  );
};

export default SystemStatus;
