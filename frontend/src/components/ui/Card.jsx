import React from 'react';

const Card = ({ 
  children, 
  title, 
  subtitle, 
  icon: Icon, 
  actions,
  className = '',
  ...props 
}) => {
  return (
    <div 
      className={`bg-white rounded-xl shadow-sm p-6 border border-gray-100 transition-all hover:shadow-md ${className}`}
      {...props}
    >
      {(title || subtitle || Icon || actions) && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {Icon && (
              <div className="p-2 bg-blue-50 rounded-lg">
                <Icon className="w-5 h-5 text-blue-600" />
              </div>
            )}
            <div>
              {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
              {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
            </div>
          </div>
          {actions && (
            <div className="flex-shrink-0">
              {actions}
            </div>
          )}
        </div>
      )}
      {children}
    </div>
  );
};

export default Card;
