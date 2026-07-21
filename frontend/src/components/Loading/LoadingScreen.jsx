import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, XCircle } from 'lucide-react';

const LoadingScreen = ({ 
  isLoading, 
  isSuccess = true, 
  message = 'Loading...',
  videoSrc = null,
  onComplete = null,
  videoDuration = 4000
}) => {
  const [progress, setProgress] = useState(0);
  const [showComplete, setShowComplete] = useState(false);
  const videoRef = useRef(null);
  const startTimeRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timeoutRef = useRef(null);

  // Dedicated effect for video changes - restarts video when src changes
  useEffect(() => {
    if (!videoRef.current || !videoSrc || !isLoading) return;

    const video = videoRef.current;
    video.pause();
    video.currentTime = 0;
    video.load();
    video.play().catch(() => {});
  }, [videoSrc, isLoading]);

  useEffect(() => {
    if (isLoading) {
      // Reset all states
      setProgress(0);
      setShowComplete(false);
      startTimeRef.current = Date.now();

      // Start progress animation
      const updateProgress = () => {
        if (!startTimeRef.current) return;
        const elapsed = Date.now() - startTimeRef.current;
        const newProgress = Math.min((elapsed / videoDuration) * 100, 100);
        setProgress(newProgress);

        if (newProgress < 100) {
          animationFrameRef.current = requestAnimationFrame(updateProgress);
        } else {
          setShowComplete(true);
          if (timeoutRef.current) clearTimeout(timeoutRef.current);
          timeoutRef.current = setTimeout(() => {
            if (onComplete) {
              onComplete();
            }
          }, 500);
        }
      };

      animationFrameRef.current = requestAnimationFrame(updateProgress);

      // Safety timeout
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        if (!showComplete) {
          setShowComplete(true);
          setTimeout(() => {
            if (onComplete) {
              onComplete();
            }
          }, 500);
        }
      }, videoDuration + 1000);

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        startTimeRef.current = null;
        if (videoRef.current) {
          videoRef.current.pause();
        }
      };
    }
  }, [isLoading, videoDuration, onComplete]);

  if (!isLoading) return null;

  const getStatusIcon = () => {
    if (showComplete) {
      return isSuccess ? 
        <CheckCircle className="w-16 h-16 text-green-500" /> : 
        <XCircle className="w-16 h-16 text-red-500" />;
    }
    return <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />;
  };

  const getStatusText = () => {
    if (showComplete) {
      return isSuccess ? 'Success!' : 'Failed';
    }
    return message;
  };

  const getStatusColor = () => {
    if (showComplete) {
      return isSuccess ? 'text-green-600' : 'text-red-600';
    }
    return 'text-white';
  };

  const getProgressColor = () => {
    if (showComplete) {
      return isSuccess ? 'bg-green-500' : 'bg-red-500';
    }
    return 'bg-blue-400';
  };

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-br from-blue-900 to-indigo-900"
        >
          <div className="flex flex-col items-center justify-center max-w-md w-full px-4">
            {videoSrc && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="relative w-56 h-56 mb-6 rounded-2xl overflow-hidden shadow-2xl border-2 border-white/10"
              >
                <video
                  key={videoSrc}
                  ref={videoRef}
                  src={videoSrc}
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-blue-900/50 via-transparent to-transparent pointer-events-none" />
                
                <svg className="absolute -top-1 -left-1 w-[calc(100%+8px)] h-[calc(100%+8px)] -rotate-90">
                  <circle
                    cx="50%"
                    cy="50%"
                    r="calc(50% - 2px)"
                    fill="none"
                    stroke="rgba(255,255,255,0.1)"
                    strokeWidth="3"
                  />
                  <circle
                    cx="50%"
                    cy="50%"
                    r="calc(50% - 2px)"
                    fill="none"
                    stroke={showComplete ? (isSuccess ? '#22c55e' : '#ef4444') : '#60a5fa'}
                    strokeWidth="3"
                    strokeDasharray={`${progress} 100`}
                    strokeLinecap="round"
                    className="transition-all duration-300"
                  />
                </svg>
              </motion.div>
            )}

            {showComplete && (
              <motion.div
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="mb-4"
              >
                {getStatusIcon()}
              </motion.div>
            )}

            <motion.h2
              initial={{ y: 10, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className={`text-2xl font-bold ${getStatusColor()} mb-2 text-center`}
            >
              {getStatusText()}
            </motion.h2>

            <div className="w-full max-w-xs bg-white/20 rounded-full h-2.5 mt-4 overflow-hidden">
              <motion.div
                className={`h-full rounded-full transition-all duration-300 ${getProgressColor()}`}
                style={{ width: `${Math.min(progress, 100)}%` }}
                initial={{ width: '0%' }}
                animate={{ width: `${Math.min(progress, 100)}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="text-white/70 text-sm mt-2"
            >
              {showComplete ? 'Complete' : `${Math.round(Math.min(progress, 100))}%`}
            </motion.p>

            {!showComplete && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.7 }}
                className="text-white/50 text-xs mt-4 text-center max-w-xs"
              >
                Please wait while we prepare your workspace...
              </motion.p>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default LoadingScreen;
