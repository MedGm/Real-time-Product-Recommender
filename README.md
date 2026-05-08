# Real-Time Product Recommender

Big Data module project · Amazon Fine Food Reviews (~568k reviews) · ALS collaborative filtering with bias decomposition

![Architecture](architecture.png)

---

## Problem

Amazon Fine Food Reviews is a textbook cold-start, sparsity, and bias problem:

- **99.9971% matrix sparsity** — most user–product pairs have no interaction
- **Extreme positive bias** — 63.9% of ratings are 5-star, median = 5; a global-mean baseline achieves RMSE ≈ 1.55 and is essentially useless
- **68.5% of users have exactly one review** — collaborative filtering degrades severely for users with one or two interactions
- **Noisy labels** — community helpfulness scores reveal that ~15% of reviews are flagged as unhelpful

The goal: build an end-to-end streaming pipeline that ingests reviews, trains a bias-aware ALS model, serves personalized Top-N recommendations in real time, and exposes a live dashboard — all containerized and reproducible.

---

## Architecture

```
Reviews.csv
    │
    ▼
┌──────────┐   JSON messages    ┌───────────┐
│ Producer │ ─────────────────► │   Kafka   │  amazon-reviews topic
│ (Airflow │                    │ (topic)   │
│  Task 1) │                    └─────┬─────┘
└──────────┘                          │
                        ┌─────────────┴──────────────┐
                        │                            │
                        ▼  spark.read (bounded)      ▼  spark.readStream (unbounded)
               ┌─────────────────┐         ┌──────────────────┐
               │   train.py      │         │   stream.py      │
               │ Spark Batch     │         │ Spark Streaming  │
               │ ALS + TVS       │         │ frozen model     │
               │ 80/10/10 split  │         │ 30s micro-batch  │
               └────────┬────────┘         └────────┬─────────┘
                        │                           │
                        └──────────┬────────────────┘
                                   ▼
                            ┌────────────┐
                            │ PostgreSQL │  recommendations + products
                            └─────┬──────┘
                                  │
                            ┌─────▼──────┐
                            │  FastAPI   │  REST + SSE
                            └─────┬──────┘
                                  │
                            ┌─────▼──────┐
                            │  Dashboard │  Nginx + HTML/JS SPA
                            └────────────┘

Orchestration: Airflow DAG (recommendation_pipeline, @once, auto-unpaused)
```

Lambda-inspired split: `train.py` reads a bounded Kafka snapshot for batch ALS training; `stream.py` consumes the same topic as an unbounded stream for real-time inference using the frozen model. Retraining triggers a `docker restart spark-stream` so the streaming job reloads the new model without downtime.

---

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Message broker | Apache Kafka 3.4 (Confluent 7.5.0) | Decouples ingestion from training; both batch and streaming consumers read the same topic independently |
| Batch processing | Apache Spark 3.5.0 (PySpark) | `spark.read.format("kafka")` for bounded training snapshot; MLlib ALS + TrainValidationSplit natively distributed |
| Stream processing | Spark Structured Streaming 3.5.0 | `spark.readStream.format("kafka")` for unbounded inference; `foreachBatch` handler with PostgreSQL upsert |
| Orchestration | Apache Airflow 2.9.1 | DAG enforces task ordering (ingest → train → reload → log); execution timeouts; single-container LocalExecutor fits the project scope |
| Storage | PostgreSQL 15 | UNIQUE constraint on `(user_id, product_id)` enables atomic upsert via staging table pattern; no separate cache layer needed |
| API | FastAPI + Uvicorn | Async Python; connection pool (psycopg2 SimpleConnectionPool); SSE endpoint for live Kafka feed |
| Frontend | Nginx + vanilla HTML/JS | No framework needed; Nginx proxies `/api/` to FastAPI and disables buffering on the SSE route; zero client-side build step |
| Containerization | Docker Compose | Reproducible single-command deployment; shared named volumes for model artifacts |

---

## System Flow

### Data lifecycle

```
1. INGEST (Airflow Task 1 — ~8 min for 200k messages)
   Reviews.csv → producer.py → Kafka topic (amazon-reviews)
   Each message: {UserId, ProductId, Score, Time,
                  HelpfulnessNumerator, HelpfulnessDenominator, Summary}

2. TRAIN (Airflow Task 2 — ~15–20 min)
   Kafka (bounded snapshot, earliest→latest)
     → drop nulls, filter rating ∈ [1,5]
     → helpfulness filter: keep h_den=0 OR h_num/h_den ≥ 0.6  (−15% noisy reviews)
     → dedup on (UserId, ProductId)
     → activity filter: MIN_USER_RATINGS=3, MIN_PRODUCT_RATINGS=3
     → StringIndexer: UserId→userId (int), ProductId→productId (int)
     → 80/10/10 split (seed=42)  ← split happens BEFORE bias computation
     → bias decomposition (train split only):
           global_mean μ = mean(train ratings)
           user_bias  b_u = mean(user ratings) − μ    per user
           item_bias  b_i = mean(item ratings) − μ    per item
           residual   ε   = rating − μ − b_u − b_i
     → ALS on residuals (nonnegative=False, coldStartStrategy=drop)
     → TrainValidationSplit: rank∈{20,50,100} × regParam∈{0.05,0.1} × maxIter=15
     → best model evaluated on held-out test set
     → write: ALSModel, encoders, user_biases, item_biases, metrics.json,
              popular_products.json, all-user recommendations → PostgreSQL

3. RELOAD (Airflow Task 3 — <5 s)
   docker restart spark-stream → stream.py reloads new model from /model volume

4. STREAM (always-on, 30s micro-batches)
   Kafka (readStream, earliest, checkpointed)
     → extract unique users per batch
     → known users: ALS recommendForUserSubset → bias correction → upsert PostgreSQL
     → cold-start users: insert popular-product fallback (ON CONFLICT DO NOTHING)

5. SERVE (FastAPI, real-time)
   GET /recommendations/user/{id} → PostgreSQL lookup → JSON
   GET /feed                      → SSE, Kafka consumer (latest), 1 event/sec rate limit
   GET /health/all                → check postgres, kafka, spark training, spark streaming
```

### Bias decomposition rationale

Training on raw ratings forces ALS latent factors to model rating scale position rather than preference structure. After decomposition, residuals are centered near zero and the latent factors capture genuine user–item affinity:

```
Naive global mean predictor:  RMSE ≈ 1.55
ALS on raw ratings:            RMSE ≈ 1.03–1.40  (varies with hyperparams)
ALS on residuals (this work):  RMSE = 0.497 (test, honest — split before bias computation)
```

**Important**: biases are computed exclusively from the training split. Computing biases on the full dataset before splitting constitutes data leakage (inflates RMSE to ~0.24, which is unrealistic).

---

## Deployment

### Prerequisites

- Docker + Docker Compose v2
- ≥ 8 GB RAM allocated to Docker
- `dataset/Reviews.csv` — download from [Kaggle: amazon-fine-food-reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

### Launch

```bash
# Full clean start — wipes all volumes, rebuilds images
docker compose down -v && docker compose up --build
```

Pipeline runs automatically. No manual Airflow trigger needed on a fresh stack.

Total time from cold start to recommendations available: **~25–35 minutes**.

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| FastAPI docs | http://localhost:8000/docs | — |
| Spark UI | http://localhost:8090 | — |
| PostgreSQL | localhost:5433 | `bigdata` / `bigdata` |

### Useful commands

```bash
# Retrain manually (e.g. after loading more data)
docker compose exec airflow airflow dags trigger recommendation_pipeline

# Clean Kafka topic without full restart
docker compose exec kafka kafka-configs --bootstrap-server localhost:29092 \
  --entity-type topics --entity-name amazon-reviews \
  --alter --add-config retention.ms=1
sleep 8
docker compose exec kafka kafka-configs --bootstrap-server localhost:29092 \
  --entity-type topics --entity-name amazon-reviews \
  --alter --delete-config retention.ms

# Rebuild a single service
docker compose up --build api
docker compose up --build dashboard

# Follow training logs
docker compose logs -f airflow
docker compose logs -f spark-stream
```

### Container topology

```
zookeeper      Confluent 7.5.0    (internal only)
kafka          Confluent 7.5.0    :9092
kafka-init     one-shot           creates topic, exits
spark-master   Spark 3.5.0        :7077, :8090
spark-worker   Spark 3.5.0        (registers with master)
spark-stream   Spark 3.5.0        always-on streaming inference
postgres       PostgreSQL 15      :5433
airflow        Airflow 2.9.1      :8080  (scheduler + webserver, LocalExecutor)
api            FastAPI + Uvicorn  :8000
dashboard      Nginx              :8501
producer       (profile: manual)  runs as Airflow Task 1, not auto-started
```

---

## Observability

### Dashboard panels

| Panel | What it shows |
|-------|--------------|
| Live Feed | SSE stream of Kafka events; review summary, product ID, score, timestamp; rate-limited to 1 event/s to prevent browser OOM |
| Recommendations | Top-N for any user; bias breakdown (μ + b_u + b_i + ε) per recommendation; cold-start detection |
| Model Analytics | Val/test RMSE; RMSE comparison vs baseline and raw ALS; hyperparameter table; data split summary |
| Dataset EDA | Full-dataset stats (568k reviews, sparsity, rating distribution); why bias decomposition was needed |
| Pipeline Health | Live status of postgres, kafka, spark training (metrics.json), spark streaming (checkpoint age); 30s auto-refresh |

### Metrics endpoint

```bash
GET /metrics
{
  "val_rmse":        0.3552,
  "test_rmse":       0.4974,
  "best_params":     {"rank": 100, "regParam": 0.05, "maxIter": 15},
  "global_mean":     4.3356,
  "trained_at":      "2026-05-08T15:03:51Z",
  "finished_at":     "2026-05-08T15:10:30Z",
  "train_rows":      39859,
  "unique_users":    11751,
  "unique_products": 3033
}
```

### Health check

```bash
GET /health/all
{
  "overall": "healthy",
  "components": [
    {"name": "postgres",          "status": "healthy",  "detail": "115,350 recommendation rows"},
    {"name": "kafka",             "status": "healthy",  "detail": "200,000 messages in amazon-reviews"},
    {"name": "spark_training",    "status": "healthy",  "detail": "test RMSE 0.4974 · finished 2026-05-08"},
    {"name": "spark_streaming",   "status": "healthy",  "detail": "last checkpoint 36s ago"}
  ]
}
```

### Spark streaming checkpoint

`stream.py` writes a Spark Structured Streaming checkpoint to `/model/stream_checkpoint`. The health monitor checks the age of the most recent checkpoint file — if > 120 seconds, status degrades to `degraded`. A wiped volume (`down -v`) triggers full Kafka replay on next start.

---

## Results

### Model performance

| Configuration | Test RMSE | Notes |
|---|---|---|
| Global mean predictor | ~1.55 | Predicts μ = 4.18 for every user-product pair |
| Raw ALS (rank∈{10,20}, reg∈{0.01,0.1}) | ~1.03–1.40 | No bias decomposition |
| **ALS + bias decomposition** | **0.4974** | Split-before-bias, train-only statistics |

RMSE is evaluated on held-out residuals (10% test split, never seen during training or hyperparameter tuning). Best hyperparameters: `rank=100, regParam=0.05, maxIter=15`.

### Scale

| Metric | Value |
|--------|-------|
| Kafka messages ingested | 200,000 |
| Rows after cleaning + dedup + activity filter | ~39,859 |
| Users with recommendations | 11,751 |
| Unique products indexed | 3,033 |
| Total recommendation rows in PostgreSQL | ~115,000–1.2M (Top-10 per user) |
| Ingestion time (200k messages, no delay) | ~8 minutes |
| Training time (6-param TVS, rank=100) | ~15–20 minutes |
| API p50 recommendation latency | < 5 ms (PostgreSQL index scan) |

---

## Limitations

**Data leakage surface**: biases are now computed from training rows only, but the StringIndexer vocabulary is still fit on the full dataset. This is a minor form of leakage (vocabulary, not statistics) and is acceptable since indexes carry no rating information.

**Single-partition Kafka topic**: the `amazon-reviews` topic uses 1 partition. Spark reads and stream consumers cannot parallelize across partitions. Throughput is constrained by single-broker I/O.

**LocalExecutor in Airflow**: the scheduler and task workers share a single container. Under concurrent DAG runs or slow training, the Airflow webserver can become unresponsive. A production setup would use CeleryExecutor or KubernetesExecutor.

**Streaming inference replays on restart**: `stream.py` uses `startingOffsets="earliest"` with checkpointing. A wiped checkpoint (e.g. `down -v`) causes full Kafka replay — 200k messages reprocessed on restart. In production, a persistent external checkpoint store (S3, GCS) would prevent this.

**Cold-start users get popular-product fallback only**: users not seen during training receive the global Top-N most-reviewed products. This ignores any session context or content features. A hybrid approach (content-based fallback using product category) would improve cold-start quality.

**No online model update**: ALS is retrained from scratch on each DAG trigger. Incremental updates (e.g. online SGD, factorization machines) are not implemented. The model is stale between retraining runs.

**Val RMSE (0.3552) < Test RMSE (0.4974)**: gap of 0.14 suggests the val set happened to be easier than the test set for this particular random split (seed=42). With only ~40k training rows, high split variance is expected. Both metrics are computed on residuals, not raw ratings.

---

## Future Improvements

**Model**
- Replace ALS with a neural collaborative filtering model (NeuMF, LightGCN) for improved accuracy on sparse data
- Add content features (product category, review text embeddings) to reduce cold-start RMSE
- Implement online learning to incrementally update user/item embeddings without full retraining

**Pipeline**
- Move checkpoint storage to S3/GCS so `down -v` does not trigger full Kafka replay
- Replace LocalExecutor with KubernetesExecutor for proper task isolation and resource limits
- Add data quality checks (Great Expectations or custom Spark assertions) before training starts
- Partition the Kafka topic (≥ 3 partitions) to enable parallel Spark consumers

**Infrastructure**
- Add Prometheus metrics export from FastAPI and Spark; Grafana dashboard for operational visibility
- Implement circuit breaker on FastAPI → PostgreSQL calls to handle DB restart gracefully
- Add authentication to FastAPI endpoints (JWT or API key) before any external exposure
- Use Kafka Schema Registry to enforce message schema and detect producer-side changes early

**Evaluation**
- Add offline ranking metrics (NDCG@10, Precision@K, Recall@K) alongside RMSE — RMSE on residuals is not directly interpretable as recommendation quality
- Implement A/B evaluation framework: compare bias-decomposed ALS vs raw ALS on held-out users
