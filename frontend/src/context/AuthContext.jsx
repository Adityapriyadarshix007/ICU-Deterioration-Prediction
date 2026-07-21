import React, { createContext, useState, useEffect, useRef } from 'react';
import api from '../services/api';
import toast from 'react-hot-toast';
import { useLoading } from './LoadingContext';
import logoutVideo from '../assets/logout.mp4';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const isLoggingIn = useRef(false);
  const { showLoading, hideLoading } = useLoading();

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        await loadUser();
      } catch (error) {
        console.error('❌ Auth initialization failed:', error);
        localStorage.removeItem('access_token');
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    initAuth();
  }, []);

  const loadUser = async () => {
    try {
      const response = await api.getCurrentUser();
      console.log('✅ User loaded:', response.data);
      const userData = response.data;
      setUser(userData);
      return userData;
    } catch (error) {
      console.error('❌ Failed to load user:', error);
      localStorage.removeItem('access_token');
      setUser(null);
      throw error;
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
      setLoading(false);
      return userData;
    } catch (error) {
      localStorage.removeItem('access_token');
      setUser(null);
      setLoading(false);
      throw error;
    } finally {
      isLoggingIn.current = false;
    }
  };

  const logout = async () => {
    // Show loading with logout animation using existing LoadingContext
    showLoading({
      isSuccess: true,
      message: 'Logging out...',
      videoSrc: logoutVideo,
      videoDuration: 4000
    });

    try {
      await api.logout();
    } catch (error) {
      // ignore
    }
    
    // Wait for the full 4-second animation to complete
    // The LoadingScreen's onComplete will call hideLoading
    // We need to clear auth state after the animation finishes
    setTimeout(() => {
      localStorage.removeItem('access_token');
      setUser(null);
      toast.success('Logged out successfully');
      // The LoadingScreen will auto-hide via onComplete callback
    }, 4000);
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
