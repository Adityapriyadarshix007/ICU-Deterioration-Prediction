import React, { createContext, useState, useEffect, useRef } from 'react';
import api from '../services/api';
import toast from 'react-hot-toast';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const isLoggingIn = useRef(false);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      loadUser();
    } else {
      setLoading(false);
    }
  }, []);

  const loadUser = async () => {
    try {
      const response = await api.getCurrentUser();
      console.log('✅ User loaded:', response.data);
      setUser(response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to load user:', error);
      localStorage.removeItem('access_token');
      setUser(null);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const login = async (token) => {
    if (isLoggingIn.current) {
      console.log('⏳ Login already in progress');
      return;
    }
    
    isLoggingIn.current = true;
    
    try {
      localStorage.setItem('access_token', token);
      const userData = await loadUser();
      console.log('✅ Login successful, user:', userData);
      return userData;
    } catch (error) {
      localStorage.removeItem('access_token');
      setUser(null);
      throw error;
    } finally {
      isLoggingIn.current = false;
    }
  };

  const logout = async () => {
    try {
      await api.logout();
    } catch (error) {
      // ignore
    }
    localStorage.removeItem('access_token');
    setUser(null);
    toast.success('Logged out successfully');
  };

  const value = {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user,
    isLoggingIn: isLoggingIn.current,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
