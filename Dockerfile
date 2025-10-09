# Base image
FROM apache/airflow:2.9.2-python3.12

# Switch to root to install system dependencies
USER root

# Install system dependencies needed for Prophet, LightGBM, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt



# Copy project code
COPY dags /opt/airflow/dags
COPY ingestion /opt/airflow/ingestion
COPY analytics /opt/airflow/analytics
COPY alerts /opt/airflow/alerts
