FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /environment

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt