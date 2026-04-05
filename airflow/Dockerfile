FROM apache/airflow:2.9.1-python3.11

USER root
# Install Java (needed for spark-submit inside Airflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Spark client (same version as spark image)
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt && \
    ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Create shared package dir (owned by airflow so pip can write to it)
RUN mkdir -p /opt/airflow-packages && chown airflow /opt/airflow-packages

# Install Python packages to a shared location accessible by all users (including root)
USER airflow
COPY requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir --target /opt/airflow-packages -r /tmp/airflow-requirements.txt

# Make packages importable by any process (spark-submit runs as root)
ENV PYTHONPATH=/opt/airflow-packages
