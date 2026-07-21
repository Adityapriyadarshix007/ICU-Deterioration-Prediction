import React, { createContext, useContext, useState, useCallback } from 'react';
import LoadingScreen from '../components/Loading/LoadingScreen';

const LoadingContext = createContext();

export const useLoading = () => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
};

export const LoadingProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingConfig, setLoadingConfig] = useState({
    isSuccess: true,
    message: 'Loading...',
    videoSrc: null,
    videoDuration: 4000,
  });

  const showLoading = useCallback((options = {}) => {
    setLoadingConfig({
      isSuccess: options.isSuccess ?? true,
      message: options.message || 'Loading...',
      videoSrc: options.videoSrc || null,
      videoDuration: options.videoDuration || 4000,
    });
    setIsLoading(true);
  }, []);

  const hideLoading = useCallback(() => {
    setIsLoading(false);
  }, []);

  const updateLoading = useCallback((options = {}) => {
    setLoadingConfig(prev => ({
      ...prev,
      isSuccess: options.isSuccess ?? prev.isSuccess,
      message: options.message || prev.message,
      videoSrc: options.videoSrc || prev.videoSrc,
      videoDuration: options.videoDuration || prev.videoDuration,
    }));
  }, []);

  return (
    <LoadingContext.Provider
      value={{
        isLoading,
        loadingConfig,
        showLoading,
        hideLoading,
        updateLoading,
      }}
    >
      {children}
      <LoadingScreen
        isLoading={isLoading}
        isSuccess={loadingConfig.isSuccess}
        message={loadingConfig.message}
        videoSrc={loadingConfig.videoSrc}
        videoDuration={loadingConfig.videoDuration}
        onComplete={hideLoading}
      />
    </LoadingContext.Provider>
  );
};

export default LoadingProvider;
