"""
Kafka producer — streams Amazon Fine Food Reviews CSV row by row.

Environment variables:
  KAFKA_BOOTSTRAP  broker address          (default: kafka:29092)
  KAFKA_TOPIC      topic name              (default: amazon-reviews)
  CSV_PATH         path to Reviews.csv     (default: /dataset/Reviews.csv)
  DELAY_MS         sleep between messages  (default: 10 ms)
  MAX_ROWS         0 = full dataset        (default: 0)
"""

import csv
import json
import logging
import os
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "amazon-reviews")
CSV_PATH        = os.getenv("CSV_PATH", "/dataset/Reviews.csv")
DELAY_S         = int(os.getenv("DELAY_MS", "10")) / 1000.0
MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))


def connect(retries: int = 10, wait: int = 5) -> KafkaProducer:
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            log.info("Connected to Kafka at %s", KAFKA_BOOTSTRAP)
            return producer
        except NoBrokersAvailable:
            log.warning("Broker not available (attempt %d/%d), retrying in %ds…", attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError("Could not connect to Kafka after %d attempts" % retries)


def stream(producer: KafkaProducer) -> None:
    sent = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if MAX_ROWS and sent >= MAX_ROWS:
                break
            msg = {
                "UserId":    row["UserId"],
                "ProductId": row["ProductId"],
                "Score":     float(row["Score"]),
                "Time":      int(row["Time"]),
            }
            producer.send(KAFKA_TOPIC, value=msg)
            sent += 1
            if sent % 10_000 == 0:
                producer.flush()
                log.info("Sent %d messages", sent)
            time.sleep(DELAY_S)

    producer.flush()
    log.info("Done. Total messages sent: %d", sent)


if __name__ == "__main__":
    p = connect()
    stream(p)
    p.close()
