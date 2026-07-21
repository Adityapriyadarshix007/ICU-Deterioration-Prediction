import React from 'react';
import Card from '../../../components/ui/Card';
import { Brain } from 'lucide-react';
import { MODEL_INFO } from '../utils/constants';

const ModelInfoCard = () => {
  return (
    <Card title="Model Information" icon={Brain}>
      <div className="space-y-4">
        <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
          <p className="text-sm text-gray-600">Current Model</p>
          <p className="text-xl font-bold text-blue-600">{MODEL_INFO.name}</p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">{MODEL_INFO.version}</span>
            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Production</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-gray-50 rounded-xl">
            <p className="text-xs text-gray-500">AUC-ROC</p>
            <p className="text-lg font-bold text-gray-900">{MODEL_INFO.aucRoc}</p>
          </div>
          <div className="p-3 bg-gray-50 rounded-xl">
            <p className="text-xs text-gray-500">Threshold</p>
            <p className="text-lg font-bold text-gray-900">{MODEL_INFO.threshold}</p>
          </div>
          <div className="p-3 bg-gray-50 rounded-xl">
            <p className="text-xs text-gray-500">Sensitivity</p>
            <p className="text-lg font-bold text-green-600">{MODEL_INFO.sensitivity}%</p>
          </div>
          <div className="p-3 bg-gray-50 rounded-xl">
            <p className="text-xs text-gray-500">Specificity</p>
            <p className="text-lg font-bold text-blue-600">{MODEL_INFO.specificity}%</p>
          </div>
        </div>

        <div className="p-3 bg-gray-50 rounded-xl">
          <p className="text-xs text-gray-500">Dataset</p>
          <p className="text-sm font-semibold text-gray-700">{MODEL_INFO.dataset}</p>
          <p className="text-xs text-gray-400">{MODEL_INFO.samples.toLocaleString()} ICU stays</p>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
          <div>
            <span className="font-medium">Features:</span> {MODEL_INFO.features}
          </div>
          <div>
            <span className="font-medium">Prediction Window:</span> {MODEL_INFO.predictionWindow}
          </div>
          <div>
            <span className="font-medium">Training:</span> {MODEL_INFO.trainingDate}
          </div>
          <div>
            <span className="font-medium">Calibration:</span> {MODEL_INFO.calibration}
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ModelInfoCard;
