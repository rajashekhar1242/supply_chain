FROM apache/airflow:2.9.2-python3.12

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libgtk-3-0 \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# Copy project code
COPY dags /opt/airflow/dags
COPY ingestion /opt/airflow/ingestion
COPY analytics /opt/airflow/analytics
COPY alerts /opt/airflow/alerts
