import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, Users, UserCog, 
  BarChart3, Settings, FileText, 
  Activity, Shield, Menu, X
} from 'lucide-react';

const AdminSidebar = () => {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const menuItems = [
    { path: '/admin', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/admin/patients', label: 'Patients', icon: Users },
    { path: '/admin/users', label: 'Users', icon: UserCog },
    { path: '/admin/analytics', label: 'Analytics', icon: BarChart3 },
    { path: '/admin/settings', label: 'Settings', icon: Settings },
    { path: '/admin/logs', label: 'Logs', icon: FileText },
  ];

  const isActive = (path) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <>
      {/* Mobile top bar with hamburger */}
      <div className="lg:hidden fixed top-0 left-0 right-0 h-14 bg-white border-b border-gray-200 flex items-center px-4 z-50">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 rounded-lg hover:bg-gray-100 transition"
          aria-label="Toggle sidebar"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
        <h2 className="ml-3 text-base font-bold text-gray-800">Admin Panel</h2>
      </div>

      {/* Overlay on mobile when open */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/40 z-40"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed left-0 top-0 h-full w-64 bg-white border-r border-gray-200 flex flex-col z-50
          transform transition-transform duration-300
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:translate-x-0
        `}
      >
        {/* Header (hidden on mobile since the top bar shows it) */}
        <div className="hidden lg:block p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-indigo-600">
          <h2 className="text-lg font-bold text-white">Admin Panel</h2>
          <p className="text-xs text-blue-200">System Management</p>
        </div>

        {/* Mobile close header */}
        <div className="lg:hidden p-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-gray-800">Admin Panel</h2>
            <p className="text-xs text-gray-500">System Management</p>
          </div>
          <button
            onClick={() => setMobileOpen(false)}
            className="p-1.5 rounded-lg hover:bg-gray-100"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.path);
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setMobileOpen(false)}
                className={`flex items-center gap-3 px-4 py-2.5 text-sm rounded-xl transition-all duration-200 ${
                  active 
                    ? 'bg-blue-50 text-blue-600 shadow-sm' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-blue-600' : 'text-gray-500'}`} />
                <span className={active ? 'font-medium' : ''}>{item.label}</span>
                {active && (
                  <span className="ml-auto w-1.5 h-6 bg-blue-600 rounded-full"></span>
                )}
              </NavLink>
            );
          })}
        </nav>
        
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center gap-3 px-4 py-2.5 text-sm text-gray-600 bg-gray-50 rounded-xl">
            <Shield className="w-4 h-4 text-gray-400" />
            <span className="text-xs">Admin Access</span>
          </div>
        </div>
      </aside>
    </>
  );
};

export default AdminSidebar;