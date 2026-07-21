#!/bin/bash
# Render-specific build script

echo "🚀 Running Render build..."
echo "Python version: $(python --version)"

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core packages first
echo "📦 Installing core packages..."
pip install --no-cache-dir -r requirements.txt

# Try to install ML packages (optional, with fallback)
echo "📦 Installing ML packages (optional)..."
pip install --no-cache-dir catboost numpy pandas scikit-learn shap matplotlib seaborn || \
pip install --no-cache-dir --no-deps catboost numpy pandas scikit-learn shap matplotlib seaborn || \
echo "⚠️ ML packages installation failed, continuing with core packages"

echo "✅ Build complete!"
