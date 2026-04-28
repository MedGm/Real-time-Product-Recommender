"""
DAG: recommendation_pipeline

Orchestrates the full pipeline in 4 tasks:
  1. wait_for_kafka_messages   — sensor that polls until the Kafka topic has
                                 ≥ MAX_ROWS messages (i.e. the producer has
                                 finished streaming the CSV)
  2. spark_batch_training      — runs train.py in local[*] mode; reads a
                                 bounded Kafka snapshot (spark.read, not
                                 spark.readStream), trains ALS 80/10/10,
                                 saves model + metrics.json to /model
  3. restart_streaming_service — issues `docker restart spark-stream` so the
                                 always-on streaming service reloads the newly
                                 saved ALS model
  4. print_model_metrics       — reads /model/metrics.json and logs RMSE to Airflow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture (Lambda-inspired split — why things are designed this way)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kafka is the ingestion layer BEFORE training:
  The Kafka producer streams Reviews.csv row-by-row to the topic. train.py
  reads a bounded snapshot of all accumulated messages using
  spark.read.format("kafka") with endingOffsets="latest". This materialises
  the interactions into a static DataFrame before ALS training begins.
  ALS remains a pure batch algorithm — it does not train on a live stream.

  Framing: "Kafka provides the streaming ingestion layer; Spark reads the
  accumulated interactions and trains ALS periodically on a bounded snapshot."

Why the producer is not a DAG task:
  The producer streams CSV row-by-row at a configurable rate (DELAY_MS).
  Tying it to a DAG task would require Airflow to wait for it to complete
  before training, which defeats streaming semantics. Instead, Docker Compose
  keeps the producer running as a service. The sensor (task 1) gates the
  pipeline on ≥ KAFKA_MIN_MSGS messages being present in the topic.
  KAFKA_MIN_MSGS is intentionally separate from MAX_ROWS (the producer size):
  training can be triggered before the producer finishes its full run.

Why spark-stream is always-on in Docker Compose, not a DAG task:
  The streaming inference job is a long-running process that never terminates.
  Making it a DAG task would block Airflow indefinitely. Docker Compose keeps
  it alive with restart: unless-stopped. After each training run, task 3
  (restart_streaming_service) issues `docker restart spark-stream` — because
  the service loads the model once at startup, a restart is needed to pick up
  the newly trained model.

spark.read vs spark.readStream:
  - train.py  uses spark.read      → bounded snapshot → static DataFrame → ALS
  - stream.py uses spark.readStream → unbounded stream → frozen ALS model inference

Schedule: manual trigger (uncomment schedule_interval for nightly retraining)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

SPARK_MASTER = os.getenv("SPARK_MASTER",    "spark://spark-master:7077")
KAFKA_BOOT   = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC  = "amazon-reviews"
MODEL_PATH   = "/model"
JOBS_PATH    = "/jobs"

# Sensor threshold — how many Kafka messages must exist before training starts.
# Intentionally SEPARATE from MAX_ROWS (producer size): even if the producer
# streams 50 000 rows we only need 10 000 confirmed before kicking off training.
# Raise this (e.g. 45000) for a fuller training set; keep low for quick demos.
# If the producer fails mid-stream, the DAG still trains once this floor is met.
KAFKA_MIN_MSGS = int(os.getenv("KAFKA_MIN_MSGS", "10000"))

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
    "owner":            "bigdata",
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}


class KafkaTopicSensor(BaseSensorOperator):
    """
    Polls until the Kafka topic has at least `min_messages` messages.

    Used to gate training on the producer having streamed enough data.
    With the default MAX_ROWS=50000 and DELAY_MS=10, the producer takes
    ~500 s — the sensor timeout is set to 1200 s to accommodate this.
    """

    def __init__(self, topic: str, bootstrap: str, min_messages: int = 1000, **kwargs):
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
    dag_id            = "recommendation_pipeline",
    default_args      = default_args,
    description       = "Kafka ingestion → Spark ALS training → restart streaming → metrics",
    schedule_interval = None,           # manual trigger (set cron for nightly retraining)
    start_date        = datetime(2024, 1, 1),
    catchup           = False,
    tags              = ["bigdata", "recommender"],
) as dag:

    # ── Task 1: Wait for enough Kafka messages (= producer finished) ──────────
    wait_kafka = KafkaTopicSensor(
        task_id       = "wait_for_kafka_messages",
        topic         = KAFKA_TOPIC,
        bootstrap     = KAFKA_BOOT,
        min_messages  = KAFKA_MIN_MSGS,   # waits for MAX_ROWS messages
        poke_interval = 30,
        timeout       = 1200,             # 20 min — covers 50k msgs @ 10ms/msg
        mode          = "poke",
    )

    # ── Task 2: Spark ALS batch training (reads bounded Kafka snapshot) ───────
    spark_train = BashOperator(
        task_id      = "spark_batch_training",
        bash_command = SPARK_SUBMIT_TRAIN + f"{JOBS_PATH}/train.py",
        env = {
            "KAFKA_BOOTSTRAP":      KAFKA_BOOT,
            "KAFKA_TOPIC":          KAFKA_TOPIC,
            "MODEL_PATH":           MODEL_PATH,
            "MIN_USER_RATINGS":     "5",
            "MIN_PRODUCT_RATINGS":  "5",
        },
        append_env = True,
    )

    # ── Task 3: Restart spark-stream so it loads the new model ───────────────
    # spark-stream has restart: unless-stopped in Docker Compose — it will
    # automatically come back up, re-run its startup script, and load the
    # newly saved /model/als + /model/encoders.
    # Requires /var/run/docker.sock mounted into the Airflow container.
    restart_stream = BashOperator(
        task_id      = "restart_streaming_service",
        bash_command = "docker restart spark-stream",
    )

    # ── Task 4: Log RMSE metrics to Airflow ───────────────────────────────────
    print_metrics = PythonOperator(
        task_id         = "print_model_metrics",
        python_callable = _print_metrics,
    )

    # ── Task dependency chain ─────────────────────────────────────────────────
    wait_kafka >> spark_train >> restart_stream >> print_metrics
