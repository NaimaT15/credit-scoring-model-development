# Base image with Python 3.9
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy application code and requirements
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose the port on which the application runs
EXPOSE 8000

# Set the command to run the API server
# Update the command as per the framework you're using
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
