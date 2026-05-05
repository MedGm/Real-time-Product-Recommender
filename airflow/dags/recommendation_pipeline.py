"""
DAG: recommendation_pipeline

Fully automated pipeline — triggers on stack start (@once) with no manual
intervention required.

Task chain:
  1. trigger_kafka_ingestion  — runs producer.py to stream Reviews.csv into Kafka
  2. spark_batch_training     — reads bounded Kafka snapshot, trains ALS 80/10/10,
                                saves model + metrics.json to /model volume
  3. restart_streaming_service— docker restart spark-stream so it reloads new model
  4. print_model_metrics      — reads /model/metrics.json and logs RMSE to Airflow

Architecture (Lambda-inspired split):
  Kafka is the streaming ingestion layer. The producer (Task 1) streams the dataset
  row-by-row into the topic. train.py (Task 2) reads a *bounded* snapshot via
  spark.read.format("kafka") — ALS remains a pure batch algorithm.
  stream.py runs as an always-on service that applies the frozen model to new events.

  spark.read      → bounded snapshot → static DataFrame → ALS training  (train.py)
  spark.readStream→ unbounded stream → frozen model inference            (stream.py)

Why spark-stream is always-on, not a DAG task:
  The streaming job never terminates. Task 3 issues `docker restart spark-stream`
  so the service reloads the newly saved model without being a long-running DAG task.

Schedule: @once — runs automatically on the first stack start.
  - Fresh stack (down -v + up --build): Airflow DB is empty → DAG runs immediately.
  - Stack restart without volume wipe: Airflow DB records prior run → no duplicate run.
  - To retrain manually: trigger via Airflow UI or `airflow dags trigger`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

KAFKA_BOOT  = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = "amazon-reviews"
MODEL_PATH  = "/model"
JOBS_PATH   = "/jobs"

SPARK_PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
    "org.postgresql:postgresql:42.7.3"
)

SPARK_SUBMIT_TRAIN = (
    f"/opt/spark/bin/spark-submit "
    f"--master local[*] "
    f"--packages {SPARK_PACKAGES} "
    f"--conf spark.sql.shuffle.partitions=50 "
    f"--conf spark.driver.memory=2g "
)

default_args = {
    "owner":            "bigdata",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


def _print_metrics(**context):
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


with DAG(
    dag_id                 = "recommendation_pipeline",
    default_args           = default_args,
    description            = "Auto: Kafka ingestion → Spark ALS training → streaming → metrics",
    schedule_interval      = "@once",        # auto-runs once on fresh stack start
    start_date             = datetime(2024, 1, 1),
    catchup                = False,
    is_paused_upon_creation= False,          # auto-unpause so DAG runs without manual intervention
    tags                   = ["bigdata", "recommender"],
) as dag:

    # ── Task 1: Kafka ingestion via producer.py ───────────────────────────────
    # Runs producer.py directly inside the Airflow container (kafka-python is
    # installed). DELAY_MS=0 → ingest at full speed, no artificial throttling.
    # Task exits only after all MAX_ROWS messages are flushed to Kafka — training
    # (Task 2) can then safely read a complete bounded snapshot.
    trigger_ingestion = BashOperator(
        task_id      = "trigger_kafka_ingestion",
        bash_command = "python3 /producer/producer.py",
        env = {
            "KAFKA_BOOTSTRAP": KAFKA_BOOT,
            "KAFKA_TOPIC":     KAFKA_TOPIC,
            "CSV_PATH":        "/dataset/Reviews.csv",
            "MAX_ROWS":        "200000",
            "DELAY_MS":        "0",     # no throttle — ingest as fast as possible
        },
        append_env      = True,
        execution_timeout = timedelta(hours=2),
    )

    # ── Task 2: Spark ALS batch training (reads bounded Kafka snapshot) ───────
    spark_train = BashOperator(
        task_id      = "spark_batch_training",
        bash_command = SPARK_SUBMIT_TRAIN + f"{JOBS_PATH}/train.py",
        env = {
            "KAFKA_BOOTSTRAP":      KAFKA_BOOT,
            "KAFKA_TOPIC":          KAFKA_TOPIC,
            "MODEL_PATH":           MODEL_PATH,
            "MIN_USER_RATINGS":     "3",
            "MIN_PRODUCT_RATINGS":  "3",
        },
        append_env        = True,
        execution_timeout = timedelta(hours=3),
    )

    # ── Task 3: Restart spark-stream to load the newly trained model ──────────
    restart_stream = BashOperator(
        task_id      = "restart_streaming_service",
        bash_command = "docker restart spark-stream",
    )

    # ── Task 4: Log RMSE metrics to Airflow ───────────────────────────────────
    print_metrics = PythonOperator(
        task_id         = "print_model_metrics",
        python_callable = _print_metrics,
    )

    # ── Dependency chain ──────────────────────────────────────────────────────
    trigger_ingestion >> spark_train >> restart_stream >> print_metrics
