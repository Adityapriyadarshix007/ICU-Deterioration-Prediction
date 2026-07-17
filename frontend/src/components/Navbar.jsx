import React, { useContext } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext.jsx';

function Navbar() {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getInitials = () => {
    if (user?.full_name) {
      return user.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    }
    return user?.username?.charAt(0).toUpperCase() || 'U';
  };

  // Check if user is admin - using multiple checks for safety
  const isAdmin = user?.is_admin === true || user?.role === 'admin';

  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/dashboard" className="text-2xl font-bold flex items-center">
              🏥 ICU Predictor
            </Link>
            <span className="text-sm bg-blue-700 px-2 py-1 rounded">v1.0</span>
          </div>
          
          <div className="flex items-center space-x-6">
            <Link to="/dashboard" className="hover:text-blue-200 transition">Dashboard</Link>
            <Link to="/patients" className="hover:text-blue-200 transition">Patients</Link>
            
            {/* Admin Link - show if user is admin */}
            {isAdmin && (
              <Link to="/admin" className="hover:text-yellow-200 transition text-yellow-300 font-medium flex items-center">
                ⚙️ Admin
                <span className="ml-1 text-xs bg-yellow-500 text-blue-900 px-1.5 py-0.5 rounded">Panel</span>
              </Link>
            )}
            
            <div className="flex items-center space-x-4 border-l border-blue-500 pl-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-blue-700 rounded-full flex items-center justify-center font-semibold text-sm">
                  {getInitials()}
                </div>
                <span className="text-sm hidden md:block">{user?.full_name || user?.username || 'User'}</span>
                {isAdmin && (
                  <span className="text-xs bg-yellow-500 text-blue-900 px-1.5 py-0.5 rounded font-bold">Admin</span>
                )}
              </div>
              <button
                onClick={handleLogout}
                className="text-sm bg-blue-700 hover:bg-blue-800 px-3 py-1 rounded-lg transition"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
