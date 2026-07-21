import React from 'react';

const LoadingSkeleton = () => (
  <div className="animate-pulse" aria-label="Loading patient statistics...">
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-3/4"></div>
        </div>
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2">
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <div className="h-6 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="space-y-4">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-10 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
      <div>
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <div className="h-6 bg-gray-200 rounded w-1/2 mb-6"></div>
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default LoadingSkeleton;
