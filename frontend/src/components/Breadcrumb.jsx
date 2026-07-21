import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';

const Breadcrumb = () => {
  const location = useLocation();
  const pathnames = location.pathname.split('/').filter(x => x);

  const getLabel = (path) => {
    const labels = {
      'admin': 'Dashboard',
      'patients': 'Patients',
      'users': 'Users',
      'analytics': 'Analytics',
      'settings': 'Settings',
      'logs': 'Logs',
      'dashboard': 'Dashboard'
    };
    return labels[path] || path.charAt(0).toUpperCase() + path.slice(1);
  };

  return (
    <nav className="flex items-center space-x-2 text-sm text-gray-600 mb-6" aria-label="Breadcrumb">
      <Link to="/dashboard" className="flex items-center hover:text-blue-600 transition-colors">
        <Home className="w-4 h-4" />
      </Link>
      {pathnames.map((value, index) => {
        const isLast = index === pathnames.length - 1;
        const to = `/${pathnames.slice(0, index + 1).join('/')}`;

        return (
          <React.Fragment key={to}>
            <ChevronRight className="w-4 h-4 text-gray-400" />
            {isLast ? (
              <span className="font-medium text-gray-900">{getLabel(value)}</span>
            ) : (
              <Link to={to} className="hover:text-blue-600 transition-colors">
                {getLabel(value)}
              </Link>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
};

export default Breadcrumb;
