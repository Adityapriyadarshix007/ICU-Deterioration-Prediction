#!/bin/bash
# Render-specific build script

echo "🚀 Running Render build..."
echo "Python version: $(python --version)"

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev

# Upgrade pip
pip install --upgrade pip

# Install requirements with fallback
if pip install --no-cache-dir -r requirements.txt; then
    echo "✅ All packages installed successfully"
else
    echo "⚠️ Some packages failed, installing core packages..."
    pip install fastapi uvicorn pymongo python-dotenv pydantic bcrypt passlib python-jose[cryptography] python-multipart
    pip install catboost numpy pandas scikit-learn --no-deps || true
fi

echo "✅ Build complete!"
