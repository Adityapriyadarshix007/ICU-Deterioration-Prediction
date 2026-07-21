import React, { useEffect, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { Loader2 } from 'lucide-react';

function OAuthCallback() {
  const navigate = useNavigate();
  const { login } = useContext(AuthContext);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const params = new URLSearchParams(window.location.search);
        const code = params.get('code');
        
        if (code) {
          console.log('OAuth callback received with code');
          setTimeout(() => {
            navigate('/dashboard');
          }, 2000);
        } else {
          navigate('/dashboard');
        }
      } catch (error) {
        console.error('OAuth callback error:', error);
        navigate('/login');
      }
    };

    handleCallback();
  }, [navigate, login]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="text-center">
        <Loader2 className="w-12 h-12 animate-spin text-blue-600 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-700">Completing Google Sign-In...</h2>
        <p className="text-gray-500 mt-2">Please wait while we redirect you.</p>
      </div>
    </div>
  );
}

export default OAuthCallback;
