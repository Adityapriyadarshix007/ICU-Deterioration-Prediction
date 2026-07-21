#!/bin/bash

echo "🚀 Deploying ICU Deterioration Prediction"

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Update deployment configuration"
git push

# Deploy backend to Render
echo "🌐 Deploying backend to Render..."
# Trigger Render deployment (automatically happens on push)

# Deploy frontend to Vercel
echo "🌐 Deploying frontend to Vercel..."
cd frontend
vercel --prod

echo "✅ Deployment complete!"
echo "🔗 Backend: https://icu-predictor-backend.onrender.com"
echo "🔗 Frontend: https://icu-predictor.vercel.app"
