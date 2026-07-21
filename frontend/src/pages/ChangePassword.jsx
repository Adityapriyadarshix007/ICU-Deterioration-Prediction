import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Mail, Lock, Eye, EyeOff, CheckCircle, 
  Loader2, ArrowLeft, Activity, HeartPulse,
  ShieldCheck, Brain
} from 'lucide-react';
import toast from 'react-hot-toast';
import api from '../services/api';
import medicalAnimation from '../assets/icon.mp4';

// Floating particles background
const Particles = () => {
  const particles = Array.from({ length: 18 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 5 + 2,
    duration: Math.random() * 15 + 12,
    delay: Math.random() * 8
  }));

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

// ECG Line
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

function ChangePassword() {
  const [email, setEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [passwordMatch, setPasswordMatch] = useState(true);
  const navigate = useNavigate();

  const validatePassword = (password) => {
    if (password.length < 8) return 'Password must be at least 8 characters';
    if (!/[A-Z]/.test(password)) return 'Must contain at least one uppercase letter';
    if (!/[a-z]/.test(password)) return 'Must contain at least one lowercase letter';
    if (!/[0-9]/.test(password)) return 'Must contain at least one number';
    return null;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    if (name === 'email') setEmail(value);
    if (name === 'newPassword') {
      setNewPassword(value);
      setPasswordMatch(value === confirmPassword);
    }
    if (name === 'confirmPassword') {
      setConfirmPassword(value);
      setPasswordMatch(newPassword === value);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate email
    if (!email) {
      toast.error('Please enter your email');
      return;
    }
    if (!/\S+@\S+\.\S+/.test(email)) {
      toast.error('Please enter a valid email address');
      return;
    }

    // Validate passwords
    if (newPassword !== confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }

    const error = validatePassword(newPassword);
    if (error) {
      toast.error(error);
      return;
    }

    setIsLoading(true);

    try {
      // Send email and new password to reset
      await api.resetPasswordByEmail(email, newPassword);
      setIsSuccess(true);
      toast.success('Password changed successfully!');
    } catch (error) {
      console.error('Change password error:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to change password. Please check your email.';
      toast.error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  if (isSuccess) {
    return (
      <div className="min-h-screen flex bg-gradient-to-br from-blue-50 via-white to-indigo-50 relative overflow-hidden">
        <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-blue-600 to-indigo-700 p-12 flex-col justify-between relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-indigo-700 to-blue-800 animate-gradient"></div>
          <Particles />
          <ECGLine />
          
          <div className="relative z-10">
            <div className="flex items-center gap-2 text-white mb-8">
              <motion.div animate={{ rotate: [0, 5, -5, 0] }} transition={{ duration: 3, repeat: Infinity }}>
                <Activity className="w-8 h-8" />
              </motion.div>
              <span className="text-xl font-bold tracking-tight">ICU Predictor</span>
              <span className="ml-2 px-2 py-0.5 text-[10px] bg-white/20 rounded-full font-normal">Research Prototype</span>
            </div>
          </div>

          <div className="relative z-10">
            <div className="flex flex-wrap items-center justify-center gap-4 text-white/60 text-xs">
              <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> CatBoost AI</span>
              <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> SHAP Explainability</span>
              <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> MIMIC-IV Dataset</span>
            </div>
            <div className="mt-4 text-center text-white/30 text-xs">© 2026 ICU Predictor · Version 1.0 · MIMIC-IV v3.1</div>
          </div>
        </div>

        <div className="w-full lg:w-1/2 flex items-center justify-center p-8 relative">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="w-full max-w-md"
          >
            <div className="lg:hidden flex items-center gap-2 text-blue-600 mb-8">
              <Activity className="w-6 h-6" />
              <span className="text-xl font-bold">ICU Predictor</span>
              <span className="text-[10px] bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">v1.0</span>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-center"
            >
              <div className="flex justify-center mb-6">
                <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center">
                  <CheckCircle className="w-10 h-10 text-green-600" />
                </div>
              </div>
              
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Password Changed!</h2>
              <p className="text-gray-600 mb-6">
                Your password has been updated successfully.
              </p>

              <button
                onClick={() => navigate('/login')}
                className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Login
              </button>
            </motion.div>
          </motion.div>
        </div>

        <style>{`
          @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
          .animate-gradient { background-size: 200% 200%; animation: gradient 15s ease infinite; }
        `}</style>
      </div>
    );
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
            <motion.div animate={{ rotate: [0, 5, -5, 0] }} transition={{ duration: 3, repeat: Infinity }}>
              <Activity className="w-8 h-8" />
            </motion.div>
            <span className="text-xl font-bold tracking-tight">ICU Predictor</span>
            <span className="ml-2 px-2 py-0.5 text-[10px] bg-white/20 rounded-full font-normal">Research Prototype</span>
          </div>
          
          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="relative w-60 h-60 xl:w-64 xl:h-64 mx-auto -mt-2 mb-4"
          >
            <div className="absolute inset-0 bg-blue-400/20 blur-3xl rounded-full"></div>
            <div className="relative w-full h-full rounded-2xl overflow-hidden">
              <video src={medicalAnimation} autoPlay loop muted playsInline className="w-full h-full object-contain" />
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
                { icon: Brain, text: 'Trained on 57,515 MIMIC-IV ICU Stays', delay: 0.6 },
                { icon: HeartPulse, text: 'Predict 12-18 Hours Before Deterioration', delay: 0.8 }
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: item.delay, duration: 0.5 }}
                  className="flex items-center gap-3 text-white/90 bg-white/5 backdrop-blur-sm p-3 rounded-xl hover:bg-white/10 hover:scale-[1.02] transition-all duration-300"
                >
                  <div className="p-2 bg-white/10 rounded-lg"><item.icon className="w-5 h-5" /></div>
                  <span className="text-sm font-medium">{item.text}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        <div className="relative z-10">
          <div className="flex flex-wrap items-center justify-center gap-4 text-white/60 text-xs">
            <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> CatBoost AI</span>
            <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> SHAP Explainability</span>
            <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> MIMIC-IV Dataset</span>
            <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> FastAPI Backend</span>
            <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-green-400" /> React Frontend</span>
          </div>
          <div className="mt-4 text-center text-white/30 text-xs">© 2026 ICU Predictor · Version 1.0 · MIMIC-IV v3.1</div>
        </div>
      </div>

      {/* Right Panel - Change Password Form */}
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
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Change Password</h2>
            <p className="text-gray-500 mb-6">
              Enter your email and new password
            </p>
          </motion.div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email Field */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  value={email}
                  onChange={handleChange}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                  placeholder="doctor@hospital.com"
                  required
                  disabled={isLoading}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Enter the email associated with your account
              </p>
            </motion.div>

            {/* New Password */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-1">
                New Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  name="newPassword"
                  value={newPassword}
                  onChange={handleChange}
                  className="w-full pl-10 pr-12 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg"
                  placeholder="Enter new password"
                  required
                  disabled={isLoading}
                  minLength={8}
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Must contain at least 8 characters, one uppercase, one lowercase, and one number
              </p>
            </motion.div>

            {/* Confirm Password */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Confirm New Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  name="confirmPassword"
                  value={confirmPassword}
                  onChange={handleChange}
                  className={`w-full pl-10 pr-12 py-2.5 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200 hover:border-blue-300 focus:shadow-lg ${
                    !passwordMatch && confirmPassword ? 'border-red-500' : 'border-gray-300'
                  }`}
                  placeholder="Confirm new password"
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {!passwordMatch && confirmPassword && (
                <p className="text-xs text-red-500 mt-1">Passwords do not match</p>
              )}
              {passwordMatch && confirmPassword && newPassword && (
                <p className="text-xs text-green-500 mt-1">✓ Passwords match</p>
              )}
            </motion.div>

            <motion.button
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 disabled:opacity-50 flex items-center justify-center gap-2 shadow-lg hover:shadow-xl"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Updating...
                </>
              ) : (
                'Change Password'
              )}
            </motion.button>

            <div className="text-center">
              <Link
                to="/login"
                className="text-sm text-blue-600 hover:underline font-medium transition-colors flex items-center justify-center gap-1"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Login
              </Link>
            </div>
          </form>
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

export default ChangePassword;