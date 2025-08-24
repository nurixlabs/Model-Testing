# Multi-stage build for React frontend and Python backend
FROM node:18-alpine AS frontend-builder

# Build React frontend
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci --only=production

COPY dashboard/ ./
RUN npm run build

# Python backend stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/dashboard/build ./dashboard/build

# Create directories for uploads and logs
RUN mkdir -p temp_uploads logs transcription_results

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/results || exit 1

# Run the application with Flask
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
