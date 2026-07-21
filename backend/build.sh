#!/bin/bash
# Build script for Render

echo "🚀 Installing Python dependencies..."
pip install --upgrade pip

# Try to install with --no-cache-dir and fallback options
pip install --no-cache-dir -r requirements.txt || \
pip install --no-cache-dir --no-deps -r requirements.txt || \
pip install fastapi uvicorn pymongo python-dotenv pydantic bcrypt passlib python-jose[cryptography]

echo "✅ Build complete!"
