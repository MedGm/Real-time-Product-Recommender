# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

Real-time product recommendation system using the Amazon Fine Food Reviews dataset (~568k reviews).
Lambda-inspired pipeline: Kafka ingestion → Spark ALS training → Airflow orchestration → FastAPI + Nginx dashboard.

Dataset: `dataset/Reviews.csv` — columns: `Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text`.
Fields used by pipeline: `UserId`, `ProductId`, `Score`, `Time`, `HelpfulnessNumerator`, `HelpfulnessDenominator`.

## Architecture

```
docker-compose.yml
├── zookeeper + kafka          # event ingestion (amazon-reviews topic)
├── kafka-init                 # creates topic; exits after completion
├── spark-master + spark-worker# Spark cluster (UI at :8090)
├── spark-stream               # always-on streaming inference service
├── postgres                   # recommendations + model_runs tables
├── airflow                    # orchestration — fully automated @once DAG
├── api (FastAPI)              # REST API :8000
├── dashboard (Nginx + HTML/JS)# static SPA :8501, proxies /api/ to FastAPI
└── producer                   # profile: manual — run by Airflow DAG Task 1
```

## Automated Pipeline (DAG: recommendation_pipeline)

Schedule: `@once` — auto-runs on fresh stack start (`is_paused_upon_creation=False`).
No manual trigger needed for fresh `down -v` + `up --build` restarts.

### DAG Task Chain

```
Task 1: trigger_kafka_ingestion    — python3 /producer/producer.py (200k messages, DELAY_MS=0)
Task 2: spark_batch_training       — spark-submit train.py (ALS 80/10/10)
Task 3: restart_streaming_service  — docker restart spark-stream (reloads new model)
Task 4: print_model_metrics        — reads /model/metrics.json, logs RMSE
```

### Why producer is Task 1, not a Docker service

Producer now runs as a synchronous DAG task. `DELAY_MS=0` sends 200k messages as fast as possible.
After Task 1 exits, all messages are in Kafka — Task 2 reads a complete bounded snapshot.
The `producer:` service in docker-compose.yml has `profiles: ["manual"]` so it does not auto-start.

### Why spark-stream is always-on, not a DAG task

Streaming inference never terminates. Making it a DAG task would block Airflow indefinitely.
Task 3 issues `docker restart spark-stream` via the Docker socket so the service reloads the new model.

## Architecture Rationale (Lambda-inspired Split)

`train.py` uses `spark.read.format("kafka")` with `endingOffsets="latest"` — **bounded snapshot**.
`stream.py` uses `spark.readStream.format("kafka")` — **unbounded stream**, runs forever.

| | `train.py` | `stream.py` |
|--|--|--|
| Spark API | `spark.read` | `spark.readStream` |
| Bounded? | Yes — stops at latest offset | No — runs forever |
| Role | Snapshot → train ALS | New events → apply frozen ALS |

## Data Preprocessing Pipeline (train.py)

### 1. Kafka ingestion
- Reads bounded snapshot from `amazon-reviews` topic
- Parses JSON: `{UserId, ProductId, Score, Time, HelpfulnessNumerator, HelpfulnessDenominator}`

### 2. Cleaning & filtering
- Drop nulls on `[raw_user, raw_product, rating]`
- Filter `rating ∈ [1, 5]`
- **Helpfulness filter**: keep reviews with `h_den == 0` (unvoted) OR `h_num/h_den >= 0.6`
  - Removes ~15% of reviews the community flagged as unhelpful
- Deduplicate on `(raw_user, raw_product)` — one rating per user-product pair
- Activity filter: `MIN_USER_RATINGS=3`, `MIN_PRODUCT_RATINGS=3`

### 3. String encoding
- `StringIndexer` maps `UserId` → integer `userId`, `ProductId` → integer `productId`
- Encoder pipeline saved to `/model/encoders`

### 4. Bias decomposition
- Decompose: `r = μ + b_u + b_i + ε`
- Compute `global_mean μ`, per-user bias `b_u`, per-item bias `b_i`
- ALS trains on **residuals** `ε = r − μ − b_u − b_i`
- Biases saved to `/model/user_biases/` and `/model/item_biases/` (parquet)
- At inference: `predicted_rating = ε_pred + μ + b_u + b_i`
- Expected RMSE gain over raw ALS: −0.2 to −0.4

### 5. ALS training (80/10/10 split)
- `TrainValidationSplit` (Spark MLlib tuning API, `trainRatio=0.8`, `parallelism=2`)
- Hyperparameter grid: `rank ∈ {20,50,100}`, `regParam ∈ {0.05,0.1}`, `maxIter=15` → 6 combinations
- Best params found: `rank=100, regParam=0.05, maxIter=15`
- `coldStartStrategy="drop"`, `nonnegative=False` (residuals can be negative)
- **Achieved RMSE: val=0.566, test=0.557**

### 6. Output artifacts
```
/model/
  als/              — saved ALSModel
  encoders/         — saved PipelineModel (StringIndexers)
  user_biases/      — parquet: (userId int, user_bias double)
  item_biases/      — parquet: (productId int, item_bias double)
  test_data/        — parquet: held-out 10% (never seen during training)
  metrics.json      — RMSE values, best params, global_mean, timestamps
  popular_products.json — Top-N most-reviewed products (cold-start fallback)
  stream_checkpoint/    — Spark Structured Streaming checkpoint
```

## Kafka Message Schema

```json
{
  "UserId": "A3SGXH7AUHU8GW",
  "ProductId": "B001E4KFG0",
  "Score": 4.0,
  "Time": 1346976000,
  "HelpfulnessNumerator": 1,
  "HelpfulnessDenominator": 1
}
```

## Streaming Inference (stream.py)

Loads at startup: encoder, ALSModel, `global_mean` (from metrics.json), `user_biases`, `item_biases` (parquet).

Per 30-second micro-batch:
1. Extract unique users from batch
2. **Known users** (in training set): ALS `recommendForUserSubset` → apply bias correction → upsert to postgres
3. **Cold-start users** (not in training): insert popular-product fallbacks with `ON CONFLICT DO NOTHING`

Bias correction in stream: `predicted_rating = als_residual + global_mean + user_bias + item_bias`

## Airflow DAG Details

- `schedule_interval='@once'`, `is_paused_upon_creation=False`
- `start_date=datetime(2024, 1, 1)` — in the past, so runs immediately on scheduler start
- `catchup=False` — does not backfill
- Producer task env: `DELAY_MS=0, MAX_ROWS=200000`
- Training task env: `MIN_USER_RATINGS=3, MIN_PRODUCT_RATINGS=3`
- `execution_timeout`: 2h for ingestion, 3h for training
- Docker socket mounted in Airflow for `docker restart spark-stream`

To retrain manually:
```bash
docker compose exec airflow airflow dags trigger recommendation_pipeline
```

## API Contract

```
GET /recommendations/user/{user_id}?n=10
→ {"user_id": "...", "recommendations": ["B001E4KFG0", ...], "predicted_ratings": [4.23, ...]}

GET /metrics
→ {"val_rmse": 0.566, "test_rmse": 0.557, "best_params": {...}, "global_mean": 4.18, ...}

GET /pipeline-status
→ {"model_ready": true, "val_rmse": 0.566, "unique_users": 11751, ...}

GET /stats
→ {"users_with_recommendations": 11751, "total_recommendation_rows": 117510}

GET /users?limit=100
→ {"users": ["A3SGXH7AUHU8GW", ...]}
```

## Docker Compose

```bash
# Full fresh start (pipeline runs automatically)
docker compose down -v && docker compose up --build

# Rebuild single service
docker compose up --build api

# Manual producer (bypasses DAG)
docker compose --profile manual up producer
```

## Key Environment Variables

| Variable | Default | Where |
|----------|---------|-------|
| `MAX_ROWS` | `200000` | producer task / DAG |
| `DELAY_MS` | `0` | producer task (DAG), `5` in manual profile |
| `MIN_USER_RATINGS` | `3` | train.py via DAG env |
| `MIN_PRODUCT_RATINGS` | `3` | train.py via DAG env |
| `TOP_N` | `10` | stream.py, train.py |

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | — |
| Airflow | http://localhost:8080 | airflow / airflow |
| FastAPI docs | http://localhost:8000/docs | — |
| Spark UI | http://localhost:8090 | — |
| PostgreSQL | localhost:5433 | bigdata / bigdata |
