# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time product recommendation system using the Amazon Fine Food Reviews dataset (~568k reviews). The pipeline is: Kafka ingestion → Spark ALS training → Airflow orchestration → FastAPI + Nginx dashboard.

Dataset: `dataset/Reviews.csv` — columns: `Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text`. Only `UserId`, `ProductId`, `Score`, and `Time` are used by the pipeline.

## Target Architecture

```
docker-compose.yml
├── kafka + zookeeper          # event ingestion
├── spark (master + 1 worker)  # ALS training + streaming
├── airflow                    # orchestration (DAGs)
├── postgres                   # store recommendations/results
├── api (FastAPI)              # GET /recommendations/user/{id}
└── dashboard (Nginx + HTML/JS) # interactive web UI (static, proxies /api/ to FastAPI)
```

### Planned directory layout

```
producer/          # Kafka producer that streams Reviews.csv rows
spark/
  jobs/
    train.py       # batch: Kafka snapshot → clean → filter → ALS fit → save model
    stream.py      # streaming: consume Kafka → apply frozen model → Top-N recs
airflow/
  dags/            # DAG that chains Kafka sensor → training → stream restart → metrics
api/               # FastAPI app
dashboard/         # Nginx + HTML/JS static dashboard
docker-compose.yml
```

## Architecture Rationale (Lambda-inspired Split)

### Kafka is the ingestion layer BEFORE training (not just streaming inference)

The Kafka producer streams `Reviews.csv` row-by-row to the `amazon-reviews` topic.
`train.py` reads a **bounded snapshot** from Kafka using `spark.read.format("kafka")`
with `endingOffsets="latest"`. This materialises all accumulated messages into a
static DataFrame before ALS training begins. ALS remains a pure batch algorithm.

**Framing:** *"Kafka provides the streaming ingestion layer. The producer injects
the historical dataset as a continuous event stream. When sufficient data has
accumulated (≥ KAFKA_MIN_MSGS messages), Spark reads a bounded snapshot and trains
ALS in batch. The streaming job then handles cold-start users via popular-product
fallbacks."*

> **Note on terminology:** This is a Lambda-inspired split (streaming ingestion +
> batch training + streaming inference), not a full Lambda Architecture. Classic
> Lambda Architecture also requires a separate serving layer that merges batch and
> speed layer outputs. Use "Lambda-inspired" in the report.

### `spark.read` vs `spark.readStream`

| | `train.py` | `stream.py` |
|--|--|--|
| Spark API | `spark.read.format("kafka")` | `spark.readStream.format("kafka")` |
| Bounded? | ✅ Yes — stops at `endingOffsets: latest` | ❌ No — runs forever |
| Role | Snapshot → train ALS | New events → apply frozen ALS |

### Why spark-stream is always-on in Docker Compose, not a DAG task

The streaming inference job is a long-running process that never terminates.
Making it a DAG task would block Airflow indefinitely. Docker Compose keeps it
alive with `restart: unless-stopped`. After each training run, Airflow's
`restart_streaming_service` task issues `docker restart spark-stream` so the
service reloads the newly saved model.

### Why the producer is not a DAG task

The producer streams CSV row-by-row at a configurable rate (DELAY_MS).
Tying it to a DAG task would require waiting for it to finish before training,
defeating streaming semantics. The Airflow sensor `wait_for_kafka_messages`
gates the pipeline on ≥ MAX_ROWS messages being present in the topic.

## Data splits

- **80%** training, **10%** validation/hyperparameter tuning, **10%** held-out test (never seen during training).
- ALS evaluation metric: **RMSE**.

## Key implementation details

### Kafka topic
- Topic name: `amazon-reviews`
- Message schema: `{"UserId": str, "ProductId": str, "Score": float, "Time": int}`

### Spark ALS (`pyspark.ml.recommendation.ALS`)
- UserId and ProductId must be integer-encoded before ALS (map string IDs to sequential ints).
- Save the trained model to a shared volume so the streaming job and API can load it.
- `coldStartStrategy="drop"` to avoid NaN RMSE on unknown users/items.

### Airflow DAG execution order
1. `wait_for_kafka_messages`   — sensor polls until ≥ `KAFKA_MIN_MSGS` messages (default 10 000; separate from `MAX_ROWS`)
2. `spark_batch_training`       — `spark-submit train.py` (reads Kafka bounded snapshot, trains ALS, writes all recs to PostgreSQL)
3. `restart_streaming_service`  — `docker restart spark-stream` via Docker socket (cold-start handler reloads popular products)
4. `print_model_metrics`        — reads `/model/metrics.json`, logs RMSE to Airflow

> **Docker socket security:** `restart_streaming_service` mounts `/var/run/docker.sock`
> into Airflow, giving it access to the host Docker daemon. This is acceptable for a
> local demo. In production, replace with a Kubernetes Job, Docker API with scoped
> permissions, or a deployment controller.

### API contract
```
GET /recommendations/user/{user_id}?n=10
→ {"user_id": "...", "recommendations": ["ProductId1", ...]}
```

## Docker Compose

Launch the full stack:
```bash
docker compose up --build
```

Tear down and remove volumes:
```bash
docker compose down -v
```

Rebuild a single service (e.g., after editing `api/`):
```bash
docker compose up --build api
```

## Spark jobs (inside the spark-master container)

Submit batch training (reads Kafka snapshot):
```bash
docker compose exec airflow spark-submit \
  --master local[*] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.3 \
  /jobs/train.py
```

Submit streaming job (long-running — normally runs as spark-stream service):
```bash
docker compose exec spark-master spark-submit \
  --master local[2] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.3 \
  /jobs/stream.py
```

## Airflow

Access the UI at `http://localhost:8080` (default credentials: `airflow` / `airflow`).

Trigger the main DAG manually:
```bash
docker compose exec airflow airflow dags trigger recommendation_pipeline
```

## API & Dashboard

- API: `http://localhost:8000` — docs at `/docs`
- Dashboard (Nginx HTML/JS): `http://localhost:8501`
