FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + libgomp1 for LightGBM
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install ALL Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
