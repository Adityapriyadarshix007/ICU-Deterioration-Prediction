#!/bin/bash
# Build script for Render

echo "🚀 Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Build complete!"
