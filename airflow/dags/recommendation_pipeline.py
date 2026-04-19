"""
DAG: recommendation_pipeline

Orchestrates the full pipeline:
  1. wait_kafka      — sensor that polls until the Kafka topic has ≥ 1 000 messages
  2. spark_train     — runs the ALS batch training job (local mode, reads CSV)
  3. print_metrics   — reads /model/metrics.json and logs RMSE to Airflow

Architectural note — producer and spark-stream as always-on services:
  The Kafka producer and the Spark streaming job run as long-lived Docker services
  (producer, spark-stream) rather than as DAG tasks. This is intentional:
  - The producer streams the CSV continuously; tying it to a DAG task would require
    waiting for it to finish before training, which defeats streaming semantics.
  - The streaming job is perpetually live; it self-gates on /model/metrics.json
    being present before starting inference.
  The DAG's role is to gate on sufficient data (KafkaTopicSensor), run batch
  training, and surface metrics — the standard Lambda-architecture split between
  batch and speed layers.

Schedule: manual trigger (or uncomment schedule_interval for nightly retraining)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

SPARK_MASTER = os.getenv("SPARK_MASTER",  "spark://spark-master:7077")
KAFKA_BOOT   = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC  = "amazon-reviews"
MODEL_PATH   = "/model"
JOBS_PATH    = "/jobs"

SPARK_PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
    "org.postgresql:postgresql:42.7.3"
)

# Training runs in local[*] mode — avoids cross-container rename failures on the
# shared Docker volume that occur when executors run in separate containers.
SPARK_SUBMIT_TRAIN = (
    f"/opt/spark/bin/spark-submit "
    f"--master local[*] "
    f"--packages {SPARK_PACKAGES} "
    f"--conf spark.sql.shuffle.partitions=50 "
    f"--conf spark.driver.memory=2g "
)

default_args = {
    "owner":          "bigdata",
    "retries":        1,
    "retry_delay":    timedelta(minutes=2),
    "email_on_failure": False,
}


class KafkaTopicSensor(BaseSensorOperator):
    """Polls until the Kafka topic has at least `min_messages` messages."""

    def __init__(self, topic: str, bootstrap: str, min_messages: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.topic        = topic
        self.bootstrap    = bootstrap
        self.min_messages = min_messages

    def poke(self, context):
        from kafka import KafkaConsumer
        from kafka.structs import TopicPartition

        consumer   = KafkaConsumer(bootstrap_servers=self.bootstrap)
        partitions = consumer.partitions_for_topic(self.topic)
        if not partitions:
            self.log.info("Topic %s not found yet.", self.topic)
            consumer.close()
            return False

        tps         = [TopicPartition(self.topic, p) for p in partitions]
        end_offsets = consumer.end_offsets(tps)
        total       = sum(end_offsets.values())
        self.log.info(
            "Topic %s has %d messages (need %d).",
            self.topic, total, self.min_messages,
        )
        consumer.close()
        return total >= self.min_messages


def _print_metrics(**context):
    metrics_file = f"{MODEL_PATH}/metrics.json"
    with open(metrics_file) as f:
        m = json.load(f)
    print("=" * 60)
    print(f"  Validation RMSE  : {m['val_rmse']:.4f}")
    print(f"  Test RMSE        : {m['test_rmse']:.4f}")
    print(f"  Best params      : {m['best_params']}")
    print(f"  Training rows    : {m.get('train_rows', 'N/A'):,}" if isinstance(m.get('train_rows'), int) else f"  Training rows    : {m.get('train_rows', 'N/A')}")
    print(f"  Unique users     : {m.get('unique_users', 'N/A')}")
    print(f"  Unique products  : {m.get('unique_products', 'N/A')}")
    print(f"  Finished at      : {m.get('finished_at', 'N/A')}")
    print("=" * 60)


with DAG(
    dag_id          = "recommendation_pipeline",
    default_args    = default_args,
    description     = "Kafka → Spark ALS training → streaming recommendations",
    schedule_interval = None,          # manual trigger (set cron for nightly)
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    tags            = ["bigdata", "recommender"],
) as dag:

    wait_kafka = KafkaTopicSensor(
        task_id       = "wait_for_kafka_messages",
        topic         = KAFKA_TOPIC,
        bootstrap     = KAFKA_BOOT,
        min_messages  = 1000,
        poke_interval = 15,
        timeout       = 600,
        mode          = "poke",
    )

    spark_train = BashOperator(
        task_id      = "spark_batch_training",
        bash_command = SPARK_SUBMIT_TRAIN + f"{JOBS_PATH}/train.py",
        env = {
            "CSV_PATH":             "/dataset/Reviews.csv",
            "MODEL_PATH":           MODEL_PATH,
            "MIN_USER_RATINGS":     "5",
            "MIN_PRODUCT_RATINGS":  "5",
        },
        append_env = True,
    )

    print_metrics = PythonOperator(
        task_id         = "print_model_metrics",
        python_callable = _print_metrics,
    )

    # Task dependency chain
    wait_kafka >> spark_train >> print_metrics
