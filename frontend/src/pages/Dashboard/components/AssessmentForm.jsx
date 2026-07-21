import React from 'react';
import Card from '../../../components/ui/Card';
import Button from '../../../components/ui/Button';
import { Stethoscope, Sparkles, User, Calendar, Hospital, FileText } from 'lucide-react';
import { CLINICAL_RANGES } from '../utils/constants';

const AssessmentForm = ({
  patientName,
  setPatientName,
  patientAge,
  setPatientAge,
  patientGender,
  setPatientGender,
  patientDiagnosis,
  setPatientDiagnosis,
  patientRoom,
  setPatientRoom,
  vitals,
  handleVitalChange,
  handlePredict,
  predicting,
  patientId
}) => {
  return (
    <Card title="Patient Assessment" icon={Stethoscope}>
      <form onSubmit={handlePredict} className="space-y-6">
        {/* Patient Demographics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="patientName" className="block text-sm font-medium text-gray-700 mb-1">
              Patient Name <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                id="patientName"
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                placeholder="John Doe"
                required
                disabled={predicting}
              />
            </div>
          </div>
          <div>
            <label htmlFor="patientId" className="block text-sm font-medium text-gray-700 mb-1">
              MRN (Medical Record Number)
            </label>
            <input
              id="patientId"
              type="text"
              value={patientId}
              className="w-full px-4 py-2.5 border border-gray-200 rounded-xl bg-gray-50 text-gray-500 font-mono"
              disabled
            />
          </div>
        </div>

        {/* Patient Demographics - Age, Gender, Room, Diagnosis */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label htmlFor="patientAge" className="block text-sm font-medium text-gray-700 mb-1">
              Age <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                id="patientAge"
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                placeholder="65"
                min={18}
                max={120}
                step="1"
                required
                disabled={predicting}
              />
            </div>
          </div>
          <div>
            <label htmlFor="patientGender" className="block text-sm font-medium text-gray-700 mb-1">
              Gender <span className="text-red-500">*</span>
            </label>
            <select
              id="patientGender"
              value={patientGender}
              onChange={(e) => setPatientGender(e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
              required
              disabled={predicting}
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div>
            <label htmlFor="patientRoom" className="block text-sm font-medium text-gray-700 mb-1">
              Room <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <Hospital className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                id="patientRoom"
                type="text"
                value={patientRoom}
                onChange={(e) => setPatientRoom(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                placeholder="ICU-101"
                required
                disabled={predicting}
              />
            </div>
          </div>
          <div>
            <label htmlFor="patientDiagnosis" className="block text-sm font-medium text-gray-700 mb-1">
              Diagnosis <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <FileText className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                id="patientDiagnosis"
                type="text"
                value={patientDiagnosis}
                onChange={(e) => setPatientDiagnosis(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                placeholder="Sepsis"
                required
                disabled={predicting}
              />
            </div>
          </div>
        </div>

        {/* Vital Signs */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Vital Signs</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(vitals).map(([key, value]) => {
              const range = CLINICAL_RANGES[key];
              if (!range) return null;
              
              // Set appropriate step values to avoid "nearest two values" error
              let step = "any"; // Allow any decimal by default
              let placeholder = range.unit || 'Value';
              let minVal = range.min * 0.5;
              let maxVal = range.max * 1.5;
              
              // Integer fields (whole numbers only)
              if (key === 'heart_rate' || key === 'sbp' || key === 'dbp' || key === 'urine_output' || key === 'fio2') {
                step = "1";
                minVal = Math.floor(range.min * 0.5);
                maxVal = Math.ceil(range.max * 1.5);
              } 
              // GCS special case (3-15, integer only)
              else if (key === 'gcs') {
                step = "1";
                minVal = 3;
                maxVal = 15;
                placeholder = "3-15";
              }
              // Decimal fields with 2 decimal places
              else if (key === 'lactate' || key === 'creatinine') {
                step = "0.01";
                // Round to 2 decimal places
                minVal = Math.round(range.min * 0.5 * 100) / 100;
                maxVal = Math.round(range.max * 1.5 * 100) / 100;
              }
              
              return (
                <div key={key}>
                  <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">
                    {range.label}
                  </label>
                  <input
                    id={key}
                    type="number"
                    name={key}
                    value={value}
                    onChange={handleVitalChange}
                    inputMode="decimal"
                    min={minVal}
                    max={maxVal}
                    step={step}
                    className="w-full px-3 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm transition-all"
                    placeholder={placeholder}
                    required
                    disabled={predicting}
                  />
                  <p className="text-xs text-gray-400 mt-1">
                    Range: {range.min}–{range.max} {range.unit}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <Button
          type="submit"
          loading={predicting}
          size="lg"
          className="w-full"
          icon={predicting ? null : Sparkles}
          disabled={predicting}
        >
          {predicting ? 'Analyzing Patient Data...' : '🩺 Analyze Clinical Risk'}
        </Button>
      </form>
    </Card>
  );
};

export default AssessmentForm;
