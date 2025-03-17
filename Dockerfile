# Use Python 3.11 since the model was trained with it
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install minimal system dependencies required for FastAPI and spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    libffi-dev libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip and install core dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel Cython

# ✅ Install essential dependencies separately for better layer caching
RUN pip3 install --no-cache-dir numpy==1.23.5 "spacy==3.8.4"

# ✅ Copy and install application dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --no-build-isolation -r requirements.txt

# ✅ Copy application files
COPY . .

# ✅ Clean up unnecessary files to keep the image small
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* ~/.cache/pip
