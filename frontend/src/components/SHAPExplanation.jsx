import React from 'react';

function SHAPExplanation({ explanation }) {
  if (!explanation) {
    return (
      <div className="text-center text-gray-500 py-8">
        <p>No SHAP explanation available</p>
      </div>
    );
  }

  const getColor = (impact) => {
    if (impact > 0) return 'text-red-600';
    return 'text-green-600';
  };

  const getBarColor = (impact) => {
    if (impact > 0) return 'bg-red-500';
    return 'bg-green-500';
  };

  const maxAbsImpact = Math.max(...explanation.top_contributors.map(c => Math.abs(c.impact)));

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">🧠 SHAP Explanation</h3>
      
      <div className="space-y-2">
        {explanation.top_contributors.map((contributor, idx) => {
          const barWidth = (Math.abs(contributor.impact) / maxAbsImpact) * 100;
          
          return (
            <div key={idx} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{contributor.feature}</span>
                <span className={getColor(contributor.impact)}>
                  {contributor.impact > 0 ? '↑' : '↓'} {Math.abs(contributor.impact).toFixed(3)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${getBarColor(contributor.impact)}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <div className="text-xs text-gray-500">
                Value: {contributor.value.toFixed(2)} ({contributor.direction} risk)
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Prediction Confidence</span>
          <span className="font-bold">
            {(explanation.prediction * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default SHAPExplanation;
