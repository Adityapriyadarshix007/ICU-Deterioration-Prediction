import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import Card from '../../../components/ui/Card';
import StatusBadge from '../../../components/ui/StatusBadge';
import Button from '../../../components/ui/Button';
import { Clock, Eye, ChevronRight } from 'lucide-react';
import { getRiskColor } from '../utils/helpers';

const RecentPredictions = ({ predictions }) => {
  const navigate = useNavigate();

  if (!predictions || predictions.length === 0) return null;

  const handleViewAll = () => {
    navigate('/patients');
  };

  const handleViewPatient = (patientId) => {
    if (patientId) {
      navigate(`/patients/${patientId}`);
    }
  };

  return (
    <Card 
      title="Recent Assessments" 
      icon={Clock}
      actions={
        <Button variant="outline" size="sm" onClick={handleViewAll}>
          View All
        </Button>
      }
    >
      <div className="space-y-3">
        {predictions.slice(0, 5).map((pred, idx) => {
          const patientId = pred.patient_id || pred.id;
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors cursor-pointer group"
              onClick={() => handleViewPatient(patientId)}
            >
              <div className="flex-1">
                <p className="font-medium text-gray-900">
                  {pred.patient_name || `Patient ${pred.patient_id}`}
                </p>
                <p className="text-xs text-gray-400">
                  {new Date(pred.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <StatusBadge 
                  status={getRiskColor(pred.alert_level)}
                  label={pred.alert_level}
                  size="sm"
                />
                <span className="text-sm font-semibold text-blue-600 min-w-[50px] text-right">
                  {Math.round((pred.risk_score || 0) * 100)}%
                </span>
                <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-blue-600 transition-colors" />
              </div>
            </motion.div>
          );
        })}
      </div>
    </Card>
  );
};

export default RecentPredictions;
