import React from 'react';
import { Loader2 } from 'lucide-react';

const variants = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700',
  secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300',
  outline: 'border-2 border-blue-600 text-blue-600 hover:bg-blue-50',
  ghost: 'text-gray-600 hover:bg-gray-100',
  danger: 'bg-red-600 text-white hover:bg-red-700',
  success: 'bg-green-600 text-white hover:bg-green-700',
};

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md',
  icon: Icon,
  loading = false,
  disabled = false,
  className = '',
  type = 'button',
  onClick,
  ...props 
}) => {
  const variantClasses = variants[variant] || variants.primary;
  const sizeClasses = sizes[size] || sizes.md;
  
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-2 font-medium rounded-xl transition-all duration-200 ${variantClasses} ${sizeClasses} ${className} ${
        disabled || loading ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-md'
      }`}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" />}
      {!loading && Icon && <Icon className="w-4 h-4" />}
      {children}
    </button>
  );
};

export default Button;
