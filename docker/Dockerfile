FROM python:3.11-slim

WORKDIR /environment

# Copy requirements.txt to the working directory
COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt