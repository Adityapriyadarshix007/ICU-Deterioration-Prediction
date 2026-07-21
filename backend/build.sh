#!/bin/bash
echo "🚀 Build started..."
echo "Python version: $(python --version)"

# Install core packages
pip install --upgrade pip setuptools wheel

# Install all requirements with dependencies
pip install --no-cache-dir -r requirements.txt

# Ensure click and other critical packages are installed
pip install --no-cache-dir click uvicorn[standard] fastapi pymongo python-dotenv pydantic bcrypt passlib python-jose[cryptography] python-multipart

# Try ML packages (optional)
pip install --no-cache-dir catboost numpy pandas scikit-learn shap matplotlib seaborn || echo "⚠️ ML packages failed, continuing..."

echo "✅ Build complete!"
