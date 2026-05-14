"""
DAG: recommendation_pipeline

A fully automated orchestration pipeline designed to trigger on stack startup.
This DAG handles the end-to-end flow from data ingestion to model deployment.

Pipeline Stages:
    1. Kafka Ingestion      : Streams Review data from CSV into Kafka.
    2. Batch Training       : Trains an ALS recommendation model using a Kafka snapshot.
    3. Service Deployment   : Restarts the streaming service to load the new model.
    4. Metrics Logging      : Extracts and logs model performance metrics (RMSE).

Architecture:
    The system follows a Lambda-like architecture. Batch training (train.py) processes 
    historical snapshots, while the streaming service (stream.py) provides real-time 
    inference. Both layers share the same model volume for consistency.

Schedule: 
    @once - Executes automatically upon initial stack deployment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

KAFKA_BOOT     = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC    = "amazon-reviews"
MODEL_PATH     = "/model"
JOBS_PATH      = "/jobs"

SPARK_PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
    "org.postgresql:postgresql:42.7.3"
)

SPARK_SUBMIT_TRAIN = (
    f"/opt/spark/bin/spark-submit "
    f"--master local[*] "
    f"--packages {SPARK_PACKAGES} "
    f"--conf spark.sql.shuffle.partitions=20 "
    f"--conf spark.driver.memory=4g "
    f"--conf spark.driver.maxResultSize=2g "
    f"--conf spark.sql.adaptive.enabled=true "
    f"--conf spark.ui.port=4040 "
)

DEFAULT_ARGS = {
    "owner":            "bigdata",
    "retries":          0,
    "email_on_failure": False,
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _print_metrics(**context):
    """Reads metrics from JSON and prints them to the Airflow task log."""
    metrics_file = f"{MODEL_PATH}/metrics.json"
    
    with open(metrics_file) as f:
        m = json.load(f)
        
    print("=" * 60)
    print(f"  Validation RMSE  : {m['val_rmse']:.4f}")
    print(f"  Test RMSE        : {m['test_rmse']:.4f}")
    print(f"  Best params      : {m['best_params']}")
    
    if isinstance(m.get("train_rows"), int):
        print(f"  Training rows    : {m['train_rows']:,}")
        
    print(f"  Unique users     : {m.get('unique_users', 'N/A')}")
    print(f"  Unique products  : {m.get('unique_products', 'N/A')}")
    print(f"  Finished at      : {m.get('finished_at', 'N/A')}")
    print("=" * 60)

# -----------------------------------------------------------------------------
# DAG Definition
# -----------------------------------------------------------------------------

with DAG(
    dag_id                  = "recommendation_pipeline",
    default_args            = DEFAULT_ARGS,
    description             = "Kafka Ingestion -> Spark ALS Training -> Streaming Deployment",
    schedule_interval       = "@once",
    start_date              = datetime(2024, 1, 1),
    catchup                 = False,
    is_paused_upon_creation = False,
    tags                    = ["bigdata", "recommender"],
) as dag:

    # -- Stage 1: Kafka Ingestion --
    # Ingests the dataset into Kafka at full speed (DELAY_MS=0).
    trigger_ingestion = BashOperator(
        task_id           = "trigger_kafka_ingestion",
        bash_command      = "python3 /producer/producer.py",
        env = {
            "KAFKA_BOOTSTRAP": KAFKA_BOOT,
            "KAFKA_TOPIC":     KAFKA_TOPIC,
            "CSV_PATH":        "/dataset/Reviews.csv",
            "MAX_ROWS":        "568454",
            "DELAY_MS":        "0",
        },
        append_env        = True,
        execution_timeout = timedelta(hours=2),
    )

    # -- Stage 2: Spark ALS Batch Training --
    # Reads the Kafka snapshot and trains the Collaborative Filtering model.
    spark_train = BashOperator(
        task_id           = "spark_batch_training",
        bash_command      = SPARK_SUBMIT_TRAIN + f"{JOBS_PATH}/train.py",
        env = {
            "KAFKA_BOOTSTRAP":     KAFKA_BOOT,
            "KAFKA_TOPIC":         KAFKA_TOPIC,
            "MODEL_PATH":          MODEL_PATH,
            "MIN_USER_RATINGS":    "3",
            "MIN_PRODUCT_RATINGS": "3",
        },
        append_env        = True,
        execution_timeout = timedelta(hours=3),
    )

    # -- Stage 3: Restart Streaming Service --
    # Triggers a restart of the spark-stream container to hot-reload the new model.
    restart_stream = BashOperator(
        task_id           = "restart_streaming_service",
        bash_command      = "docker restart spark-stream",
    )

    # -- Stage 4: Log Model Metrics --
    # Parses the training output and logs performance results to the console.
    print_metrics = PythonOperator(
        task_id           = "print_model_metrics",
        python_callable   = _print_metrics,
    )

    # -- Pipeline Dependencies --
    trigger_ingestion >> spark_train >> restart_stream >> print_metrics
