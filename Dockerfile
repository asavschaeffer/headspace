FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
# Using lightweight requirements for faster builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the modular application
COPY headspace/ ./headspace/

# Copy supporting modules
COPY embeddings_engine.py .
COPY tag_engine.py .
COPY data_models.py .
COPY llm_chunker.py .
COPY config_manager.py .
COPY task_queue.py .
COPY model_monitor.py .

# Copy configuration files
COPY loom_config.json .
COPY tag_keywords.json .
COPY env.example .

# Copy frontend and documents folder structure
COPY static/ ./static/
RUN mkdir -p documents

# Create volume mount point for database
VOLUME /app/data

# Expose port for headspace web server
EXPOSE 8000

# Start the headspace system using new entry point
CMD ["python", "headspace/main.py"]