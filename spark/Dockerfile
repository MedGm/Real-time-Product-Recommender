FROM apache/spark:3.5.0

USER root

# Install Python 3 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# PostgreSQL JDBC driver (needed by stream.py)
RUN curl -fsSL https://jdbc.postgresql.org/download/postgresql-42.7.3.jar \
    -o /opt/spark/jars/postgresql-42.7.3.jar

# Python dependencies for the Spark jobs
RUN pip3 install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    kafka-python==2.0.2 \
    numpy==1.24.4

ENV PYSPARK_PYTHON=python3

# Keep root so executors can write to shared /model volume
USER root
