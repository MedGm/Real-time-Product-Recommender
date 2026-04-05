"""
DAG: recommendation_pipeline

Orchestrates the full pipeline:
  1. wait_kafka      — sensor that waits until the Kafka topic has messages
  2. spark_train     — runs the ALS batch training job
  3. spark_stream    — submits the Structured Streaming job (non-blocking submit)
  4. print_metrics   — reads /model/metrics.json and logs RMSE to Airflow

Schedule: manual trigger (or change to a cron for nightly retraining)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

SPARK_MASTER  = os.getenv("SPARK_MASTER",  "spark://spark-master:7077")
KAFKA_BOOT    = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC   = "amazon-reviews"
MODEL_PATH    = "/model"
JOBS_PATH     = "/jobs"

SPARK_PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
    "org.postgresql:postgresql:42.7.3"
)

# Training runs in local mode (driver + executor in same container) to avoid
# cross-container rename failures on the shared Docker volume.
SPARK_SUBMIT_TRAIN = (
    f"/opt/spark/bin/spark-submit "
    f"--master local[*] "
    f"--packages {SPARK_PACKAGES} "
    f"--conf spark.sql.shuffle.partitions=50 "
)

# Streaming still submits to the cluster (long-running, needs worker resources)
SPARK_SUBMIT_STREAM = (
    f"/opt/spark/bin/spark-submit "
    f"--master {SPARK_MASTER} "
    f"--packages {SPARK_PACKAGES} "
    f"--conf spark.driver.host=airflow "
)

default_args = {
    "owner": "bigdata",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


class KafkaTopicSensor(BaseSensorOperator):
    """Polls until the Kafka topic has at least `min_messages` messages."""

    @apply_defaults
    def __init__(self, topic: str, bootstrap: str, min_messages: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.topic     = topic
        self.bootstrap = bootstrap
        self.min_messages = min_messages

    def poke(self, context):
        from kafka import KafkaConsumer
        from kafka.structs import TopicPartition

        consumer = KafkaConsumer(bootstrap_servers=self.bootstrap)
        partitions = consumer.partitions_for_topic(self.topic)
        if not partitions:
            self.log.info("Topic %s not found yet.", self.topic)
            consumer.close()
            return False

        tps = [TopicPartition(self.topic, p) for p in partitions]
        end_offsets = consumer.end_offsets(tps)
        total = sum(end_offsets.values())
        self.log.info("Topic %s has %d messages (need %d).", self.topic, total, self.min_messages)
        consumer.close()
        return total >= self.min_messages


def _print_metrics(**context):
    metrics_file = f"{MODEL_PATH}/metrics.json"
    with open(metrics_file) as f:
        m = json.load(f)
    print("=" * 50)
    print(f"  Validation RMSE : {m['val_rmse']:.4f}")
    print(f"  Test RMSE       : {m['test_rmse']:.4f}")
    print(f"  Best params     : {m['best_params']}")
    print("=" * 50)


with DAG(
    dag_id="recommendation_pipeline",
    default_args=default_args,
    description="Kafka → Spark ALS → Streaming recommendations",
    schedule_interval=None,          # manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bigdata", "recommender"],
) as dag:

    wait_kafka = KafkaTopicSensor(
        task_id="wait_for_kafka_messages",
        topic=KAFKA_TOPIC,
        bootstrap=KAFKA_BOOT,
        min_messages=1000,
        poke_interval=15,
        timeout=600,
        mode="poke",
    )

    spark_train = BashOperator(
        task_id="spark_batch_training",
        bash_command=(
            SPARK_SUBMIT_TRAIN +
            f"{JOBS_PATH}/train.py"
        ),
        env={
            "CSV_PATH":   "/dataset/Reviews.csv",
            "MODEL_PATH": MODEL_PATH,
        },
        append_env=True,
    )

    print_metrics = PythonOperator(
        task_id="print_model_metrics",
        python_callable=_print_metrics,
    )

    # Streaming job runs as a background process (non-blocking).
    # A real production setup would use a DockerOperator or a dedicated service.
    wait_kafka >> spark_train >> print_metrics
