import React, { useContext, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { AuthContext } from '../context/AuthContext.jsx';

function Navbar() {
  const { user, logout, isAuthenticated } = useContext(AuthContext);
  const [mobileOpen, setMobileOpen] = useState(false);

  if (!isAuthenticated) return null;

  const handleLogout = () => {
    setMobileOpen(false);
    logout();
  };

  const getInitials = () => {
    if (user?.full_name) {
      return user.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    }
    return user?.username?.charAt(0).toUpperCase() || 'U';
  };

  const isAdmin = user?.is_admin === true || user?.role === 'admin';

  const navLinks = (
    <>
      <Link to="/dashboard" onClick={() => setMobileOpen(false)} className="hover:text-blue-200 transition">Dashboard</Link>
      <Link to="/patients" onClick={() => setMobileOpen(false)} className="hover:text-blue-200 transition">Patients</Link>
      {isAdmin && (
        <Link to="/admin" onClick={() => setMobileOpen(false)} className="hover:text-yellow-200 transition text-yellow-300 font-medium flex items-center">
          ⚙️ Admin
          <span className="ml-1 text-xs bg-yellow-500 text-blue-900 px-1.5 py-0.5 rounded">Panel</span>
        </Link>
      )}
    </>
  );

  return (
    <nav className="bg-blue-600 text-white shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Brand */}
          <Link to="/dashboard" className="text-xl md:text-2xl font-bold flex items-center">
            🏥 ICU Predictor
            <span className="ml-2 hidden sm:inline text-xs bg-blue-700 px-2 py-1 rounded">v1.0</span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center space-x-6">
            {navLinks}
            <div className="flex items-center space-x-4 border-l border-blue-500 pl-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-blue-700 rounded-full flex items-center justify-center font-semibold text-sm">
                  {getInitials()}
                </div>
                <span className="text-sm hidden lg:block">{user?.full_name || user?.username || 'User'}</span>
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

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg hover:bg-blue-700 transition"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile menu */}
        {mobileOpen && (
          <div className="md:hidden mt-3 pt-3 border-t border-blue-500 space-y-3">
            <div className="flex flex-col space-y-2 text-base">
              {navLinks}
            </div>
            <div className="flex items-center justify-between pt-3 border-t border-blue-500">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-blue-700 rounded-full flex items-center justify-center font-semibold text-sm">
                  {getInitials()}
                </div>
                <div>
                  <div className="text-sm font-medium">{user?.full_name || user?.username || 'User'}</div>
                  {isAdmin && <div className="text-xs text-yellow-300">Admin</div>}
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="text-sm bg-blue-700 hover:bg-blue-800 px-3 py-1.5 rounded-lg transition"
              >
                Logout
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}

export default Navbar;