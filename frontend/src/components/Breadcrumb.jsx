import React from 'react';
import { Link, useLocation } from 'react-router-dom';

function Breadcrumb() {
  const location = useLocation();
  const pathnames = location.pathname.split('/').filter((x) => x);

  // Map path to display names
  const breadcrumbMap = {
    'admin': 'Admin',
    'patients': 'Patients',
    'users': 'Users',
    'analytics': 'Analytics',
    'settings': 'Settings',
    'logs': 'Activity Logs',
  };

  return (
    <nav className="text-sm text-gray-500 mb-4">
      <ol className="list-none p-0 inline-flex items-center space-x-2">
        <li>
          <Link to="/dashboard" className="text-blue-600 hover:text-blue-800">
            🏠 Dashboard
          </Link>
        </li>
        {pathnames.map((value, index) => {
          const to = `/${pathnames.slice(0, index + 1).join('/')}`;
          const isLast = index === pathnames.length - 1;
          const displayName = breadcrumbMap[value] || value.charAt(0).toUpperCase() + value.slice(1);

          return (
            <li key={to} className="flex items-center space-x-2">
              <span className="text-gray-400">/</span>
              {isLast ? (
                <span className="text-gray-700 font-medium">{displayName}</span>
              ) : (
                <Link to={to} className="text-blue-600 hover:text-blue-800">
                  {displayName}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default Breadcrumb;
