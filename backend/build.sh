#!/bin/bash
echo "🚀 Build started..."
echo "Python version: $(python --version)"

# Install core packages with all dependencies
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir fastapi uvicorn[standard] pymongo python-dotenv pydantic bcrypt passlib python-jose[cryptography] python-multipart click email-validator httpx

# Try ML packages (optional)
pip install --no-cache-dir catboost numpy pandas scikit-learn shap matplotlib seaborn || echo "⚠️ ML packages failed, continuing..."

echo "✅ Build complete!"
