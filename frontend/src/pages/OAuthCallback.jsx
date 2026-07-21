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
          // The token is handled by the Google OAuth library
          // Just wait briefly then navigate
          setTimeout(() => {
            navigate('/dashboard');
          }, 2000);
        } else {
          // Check if we have a credential in the URL fragment
          const hashParams = new URLSearchParams(window.location.hash.substring(1));
          const accessToken = hashParams.get('access_token');
          
          if (accessToken) {
            // Handle implicit flow
            console.log('Access token found in hash');
            // The Google OAuth library handles this
            setTimeout(() => {
              navigate('/dashboard');
            }, 2000);
          } else {
            // No token found, redirect to login
            navigate('/login');
          }
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
