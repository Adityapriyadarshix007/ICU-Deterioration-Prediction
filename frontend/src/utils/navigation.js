// Navigation utility to handle redirects without causing flicker
export const navigateTo = (navigate, path, options = {}) => {
  const { replace = true, state = {} } = options;
  navigate(path, { replace, state });
};

// Check if user is authenticated and redirect if needed
export const redirectIfAuthenticated = (isAuthenticated, navigate, targetPath = '/dashboard') => {
  if (isAuthenticated) {
    navigate(targetPath, { replace: true });
    return true;
  }
  return false;
};

// Check if user is NOT authenticated and redirect to login
export const redirectIfUnauthenticated = (isAuthenticated, navigate, targetPath = '/login') => {
  if (!isAuthenticated) {
    navigate(targetPath, { replace: true });
    return true;
  }
  return false;
};
