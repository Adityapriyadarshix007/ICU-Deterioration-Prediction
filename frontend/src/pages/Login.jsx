import React, { useState, useContext, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useAnimation, useInView } from 'framer-motion';
import { GoogleLogin } from '@react-oauth/google';
import CountUp from 'react-countup';
import toast from 'react-hot-toast';
import { AuthContext } from '../context/AuthContext';
import { useLoading } from '../context/LoadingContext';
import api from '../services/api';
import { 
  Mail, Lock, User, Eye, EyeOff, 
  ArrowRight, Activity, HeartPulse,
  ShieldCheck, Brain, Database,
  CheckCircle, Loader2, Key
} from 'lucide-react';
import medicalAnimation from '../assets/icon.mp4';
import lockVideo from '../assets/lock.mp4';
import unlockVideo from '../assets/unlock.mp4';

const Particles = () => {
  const particles = useMemo(() => 
    Array.from({ length: 18 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 5 + 2,
      duration: Math.random() * 15 + 12,
      delay: Math.random() * 8
    }))
  , []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full bg-white/20"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: particle.size,
            height: particle.size,
          }}
          animate={{
            y: [0, -40, 0],
            x: [0, 15, 0],
            opacity: [0.2, 0.8, 0.2],
            scale: [1, 1.4, 1],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            delay: particle.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
};

const ECGLine = () => (
  <div className="absolute bottom-0 left-0 right-0 h-12 opacity-10">
    <svg viewBox="0 0 1000 50" className="w-full h-full">
      <motion.polyline
        points="0,25 50,25 60,5 70,45 80,25 200,25 250,25 260,10 270,40 280,25 400,25 450,25 460,5 470,45 480,25 600,25 650,25 660,15 670,35 680,25 800,25 850,25 860,5 870,45 880,25 1000,25"
        fill="none"
        stroke="white"
        strokeWidth="2"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 3, ease: "easeInOut" }}
      />
    </svg>
  </div>
);

function Login() {
  const [isSignup, setIsSignup] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    email: '',
    full_name: '',
    confirmPassword: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [passwordMatch, setPasswordMatch] = useState(true);
  const { login, isAuthenticated } = useContext(AuthContext);
  const { showLoading, updateLoading, hideLoading, isLoading } = useLoading();
  const navigate = useNavigate();
  const controls = useAnimation();
  const ref = useRef(null);
  const inView = useInView(ref);

  useEffect(() => {
    if (inView) {
      controls.start('visible');
    }
  }, [controls, inView]);

  // If already authenticated, redirect immediately
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    if (name === 'confirmPassword' || name === 'password') {
      const password = name === 'password' ? value : formData.password;
      const confirmPassword = name === 'confirmPassword' ? value : formData.confirmPassword;
      if (name === 'password') {
        setPasswordMatch(value === formData.confirmPassword);
      } else {
        setPasswordMatch(formData.password === value);
      }
    }
  };

  const validatePassword = (password) => {
    if (password.length < 8) {
      return 'Password must be at least 8 characters long';
    }
    if (!/[A-Z]/.test(password)) {
      return 'Password must contain at least one uppercase letter';
    }
    if (!/[a-z]/.test(password)) {
      return 'Password must contain at least one lowercase letter';
    }
    if (!/[0-9]/.test(password)) {
      return 'Password must contain at least one number';
    }
    return null;
  };

  const getErrorMessage = (error) => {
    return error.response?.data?.detail ||
           error.response?.data?.message ||
           error.response?.data?.error ||
           "Something went wrong. Please try again.";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (isSignup) {
      if (formData.password !== formData.confirmPassword) {
        toast.error('Passwords do not match');
        return;
      }
      
      const passwordError = validatePassword(formData.password);
      if (passwordError) {
        toast.error(passwordError);
        return;
      }
    }
    
    setIsSubmitting(true);
    
    showLoading({
      isSuccess: true,
      message: 'Authenticating...',
      videoSrc: lockVideo,
      videoDuration: 4000
    });

    try {
      if (isSignup) {
        const registerData = {
          username: formData.username,
          email: formData.email,
          password: formData.password,
          full_name: formData.full_name
        };
        await api.register(registerData);
        toast.success('Account created! Please login.');
        setIsSignup(false);
        setFormData({ username: '', password: '', email: '', full_name: '', confirmPassword: '' });
        hideLoading();
        setIsSubmitting(false);
        return;
      }

      const response = await api.login(formData.username, formData.password);
      
      updateLoading({
        isSuccess: true,
        message: 'Access Granted!',
        videoSrc: lockVideo,
        videoDuration: 4000
      });
      
      // Perform login
      await login(response.data.access_token);
      
      // Navigate immediately after login, no waiting for useEffect
      setTimeout(() => {
        hideLoading();
        navigate('/dashboard', { replace: true });
        setIsSubmitting(false);
      }, 4000);
      
    } catch (error) {
      console.error('Login error:', error);
      
      updateLoading({
        isSuccess: false,
        message: 'Access Denied',
        videoSrc: unlockVideo,
        videoDuration: 4000
      });
      
      setTimeout(() => {
        hideLoading();
        toast.error(getErrorMessage(error));
        setIsSubmitting(false);
      }, 4000);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setIsSubmitting(true);
    
    showLoading({
      isSuccess: true,
      message: 'Authenticating with Google...',
      videoSrc: lockVideo,
      videoDuration: 4000
    });
    
    try {
      const response = await api.googleLogin(credentialResponse.credential);
      
      updateLoading({
        isSuccess: true,
        message: 'Google Access Granted!',
        videoSrc: lockVideo,
        videoDuration: 4000
      });
      
      await login(response.data.access_token);
      
      // Navigate immediately
      setTimeout(() => {
        hideLoading();
        navigate('/dashboard', { replace: true });
        setIsSubmitting(false);
      }, 4000);
      
    } catch (error) {
      console.error('Google login error:', error);
      
      updateLoading({
        isSuccess: false,
        message: 'Google Login Failed',
        videoSrc: unlockVideo,
        videoDuration: 4000
      });
      
      setTimeout(() => {
        hideLoading();
        toast.error(getErrorMessage(error));
        setIsSubmitting(false);
      }, 4000);
    }
  };

  const handleGoogleError = () => {
    toast.error('Google login failed. Please try again.');
  };

  const togglePasswordVisibility = () => setShowPassword(prev => !prev);
  const toggleConfirmPasswordVisibility = () => setShowConfirmPassword(prev => !prev);

  // If loading screen is showing or already authenticated, render nothing
  if (isLoading || isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-blue-50 via-white to-indigo-50 relative overflow-hidden">
      {/* Left Panel - Hero Section */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-blue-600 to-indigo-700 p-12 flex-col justify-between relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-indigo-700 to-blue-800 animate-gradient"></div>
        <Particles />
        <ECGLine />
        
        <div className="relative z-10">
          <div className="flex items-center gap-2 text-white mb-8">
            <motion.div
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <Activity className="w-8 h-8" />
            </motion.div>
            <span className="text-xl font-bold tracking-tight">ICU Predictor</span>
            <span className="ml-2 px-2 py-0.5 text-[10px] bg-white/20 rounded-full font-normal">
              Research Prototype
            </span>
          </div>
          
          <motion.div
            animate={{
              y: [0, -8, 0]
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="relative w-60 h-60 xl:w-64 xl:h-64 mx-auto -mt-2 mb-4"
          >
            <div className="absolute inset-0 bg-blue-400/20 blur-3xl rounded-full"></div>
            <div className="relative w-full h-full rounded-2xl overflow-hidden">
              <video
                src={medicalAnimation}
                autoPlay
                loop
                muted
                playsInline
                className="w-full h-full object-contain"
              />
            </div>
          </motion.div>
          
          <div className="space-y-6 text-center">
            <h1 className="text-4xl font-bold text-white leading-tight">
              Early ICU Deterioration<br />
              <span className="text-blue-200">AI Prediction Platform</span>
            </h1>
            
            <div className="space-y-3 text-left max-w-md mx-auto">
              {[
                { icon: Brain, text: 'AI-Powered Clinical Decision Support', delay: 0.2 },
                { icon: ShieldCheck, text: 'CatBoost Model with SHAP Explainability', delay: 0.4 },
                { icon: Database, text: 'Trained on 57,515 MIMIC-IV ICU Stays', delay: 0.6 },
                { icon: HeartPulse, text: 'Predict 12-18 Hours Before Deterioration', delay: 0.8 }
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: item.delay, duration: 0.5 }}
                  className="flex items-center gap-3 text-white/90 bg-white/5 backdrop-blur-sm p-3 rounded-xl hover:bg-white/10 hover:scale-[1.02] transition-all duration-300"
                >
                  <div className="p-2 bg-white/10 rounded-lg">
                    <item.icon className="w-5 h-5" />
                  </div>
                  <span className="text-sm font-medium">{item.text}</span>
                </motion.div>
              ))}
            </div>

            <motion.div 
              ref={ref}
              initial="hidden"
              animate={controls}
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0 }
              }}
              className="flex items-center justify-center gap-8 pt-6 border-t border-white/10"
            >
              <div className="text-center">
                <p className="text-3xl font-bold text-white">
                  <CountUp start={0} end={70.13} duration={2.5} decimals={2} suffix="%" />
                </p>
                <p className="text-sm text-blue-200 font-medium">AUC-ROC</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold text-white">
                  <CountUp start={0} end={68.47} duration={2.5} decimals={2} suffix="%" />
                </p>
                <p className="text-sm text-blue-200 font-medium">Sensitivity</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold text-white">
                  <CountUp start={0} end={62.03} duration={2.5} decimals={2} suffix="%" />
                </p>
                <p className="text-sm text-blue-200 font-medium">Specificity</p>
              </div>
            </motion.div>
          </div>
        </div>

        <div className="relative z-10">
          <div className="flex flex-wrap items-center justify-center gap-4 text-white/60 text-xs">
            <span className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-400" /> CatBoost AI
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-400" /> SHAP Explainability
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-400" /> MIMIC-IV Dataset
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-400" /> FastAPI Backend
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-400" /> React Frontend
            </span>
          </div>
          <div className="mt-4 text-center text-white/30 text-xs">
            © 2026 ICU Predictor · Version 1.0 · MIMIC-IV v3.1
          </div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full lg:w-1/2 flex items-center justify-center p-8 relative"
      >
        <div className="w-full max-w-md">
          <div className="lg:hidden flex items-center gap-2 text-blue-600 mb-8">
            <Activity className="w-6 h-6" />
            <span className="text-xl font-bold">ICU Predictor</span>
            <span className="text-[10px] bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">v1.0</span>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {isSignup ? 'Create Account' : 'Welcome Back'}
            </h2>
            <p className="text-gray-500 mb-6">
              {isSignup ? 'Start using the ICU prediction platform' : 'Sign in to continue to your dashboard'}
            </p>
          </motion.div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignup && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Full Name
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      type="text"
                      name="full_name"
                      value={formData.full_name}
                      onChange={handleChange}
                      className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                      placeholder="Dr. John Doe"
                      required
                    />
                  </div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Email
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                      placeholder="doctor@hospital.com"
                      required
                    />
                  </div>
                </motion.div>
              </>
            )}

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: isSignup ? 0.5 : 0.3 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {isSignup ? 'Username' : 'Username or Email'}
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                  placeholder={isSignup ? 'Choose a username' : 'Enter your username or email'}
                  required
                />
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: isSignup ? 0.6 : 0.4 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="w-full pl-10 pr-12 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                  placeholder={isSignup ? 'Create a strong password' : 'Enter your password'}
                  required
                  minLength={8}
                />
                <button
                  type="button"
                  onClick={togglePasswordVisibility}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {isSignup && (
                <p className="text-xs text-gray-500 mt-1">
                  Must contain at least 8 characters, one uppercase, one lowercase, and one number
                </p>
              )}
            </motion.div>

            {isSignup && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
              >
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={`w-full pl-10 pr-12 py-2.5 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg ${
                      !passwordMatch && formData.confirmPassword ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="Confirm your password"
                    required
                  />
                  <button
                    type="button"
                    onClick={toggleConfirmPasswordVisibility}
                    aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {!passwordMatch && formData.confirmPassword && (
                  <p className="text-xs text-red-500 mt-1">Passwords do not match</p>
                )}
                {passwordMatch && formData.confirmPassword && formData.password && (
                  <p className="text-xs text-green-500 mt-1">✓ Passwords match</p>
                )}
              </motion.div>
            )}

            {!isSignup && (
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => navigate('/change-password')}
                  className="text-sm text-blue-600 hover:underline font-medium transition-colors flex items-center gap-1"
                >
                  <Key className="w-3 h-3" />
                  Change Password
                </button>
              </div>
            )}

            <motion.button
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={isSubmitting || (isSignup && !passwordMatch)}
              className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 disabled:opacity-50 flex items-center justify-center gap-2 shadow-lg hover:shadow-xl"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  {isSignup ? 'Create Account' : 'Sign In'}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </motion.button>
          </form>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-white text-gray-500">or continue with</span>
            </div>
          </div>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={handleGoogleError}
              theme="outline"
              size="large"
              shape="pill"
              text="signin_with"
              width="300"
            />
          </div>

          <p className="text-center text-sm text-gray-500 mt-6">
            {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
            <button
              type="button"
              onClick={() => {
                setIsSignup(!isSignup);
                setFormData({ username: '', password: '', email: '', full_name: '', confirmPassword: '' });
                setPasswordMatch(true);
              }}
              className="text-blue-600 hover:underline font-medium transition-colors"
            >
              {isSignup ? 'Sign In' : 'Sign Up'}
            </button>
          </p>
        </div>
      </motion.div>

      <style>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% 200%;
          animation: gradient 15s ease infinite;
        }
      `}</style>
    </div>
  );
}

export default Login;