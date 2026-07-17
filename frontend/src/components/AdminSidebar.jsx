import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';

function AdminSidebar() {
  const navigate = useNavigate();

  const navItems = [
    { path: '/admin', icon: '📊', label: 'Dashboard' },
    { path: '/admin/patients', icon: '👤', label: 'Patients' },
    { path: '/admin/users', icon: '👥', label: 'Users' },
    { path: '/admin/analytics', icon: '📈', label: 'Analytics' },
    { path: '/admin/settings', icon: '⚙️', label: 'Settings' },
    { path: '/admin/logs', icon: '📋', label: 'Activity Logs' },
  ];

  return (
    <div className="w-64 bg-gray-800 min-h-screen text-white fixed left-0 top-0 pt-16 flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-bold">Admin Panel</h2>
        <p className="text-xs text-gray-400">System Management</p>
      </div>
      
      <nav className="p-4 flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-4 py-3 rounded-lg transition ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`
            }
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Back to Dashboard button */}
      <div className="p-4 border-t border-gray-700">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center space-x-3 w-full px-4 py-3 rounded-lg text-gray-300 hover:bg-gray-700 hover:text-white transition"
        >
          <span>🏠</span>
          <span>Back to Dashboard</span>
        </button>
      </div>
    </div>
  );
}

export default AdminSidebar;
