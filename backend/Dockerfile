# Use Python 3.11 explicitly
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building ML packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .

# Install core packages first
RUN pip install --no-cache-dir fastapi uvicorn pymongo python-dotenv pydantic bcrypt passlib python-jose[cryptography] python-multipart

# Try to install ML packages (catboost, numpy, pandas, etc.)
# If they fail, the app will still work with core functionality
RUN pip install --no-cache-dir catboost numpy pandas scikit-learn shap matplotlib seaborn || \
    echo "ML packages failed to install, continuing..."

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
