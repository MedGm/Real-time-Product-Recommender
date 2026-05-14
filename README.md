# Real-Time Product Recommender

**Big Data Module · Université Abdelmalek Essaâdi**  
Supervised by **Prof. Yassin El Yusfi**

Amazon Fine Food Reviews (~568k reviews) · ALS collaborative filtering with bias decomposition · End-to-end streaming pipeline

![Architecture](architecture.png)

---

## Team

| Member | Role |
|--------|------|
| **El Gorrim Mohamed** | ML Engineer — bias decomposition, ALS training pipeline, data leakage fix, PostgreSQL upsert pattern |
| **Kchibal Ismail** | Data Engineer — Kafka/ZooKeeper setup, producer, Spark Structured Streaming inference, Docker health checks |
| **Uthman Junaid** | Full-Stack Engineer — Airflow DAG, FastAPI REST API, live feed, dashboard SPA, Nginx reverse proxy |

---

## Problem Statement

The Amazon Fine Food Reviews dataset presents three compounding challenges for a recommender system:

- **Extreme sparsity** — 99.9971% of user–product pairs have no interaction (256k users × 74k products)
- **Positive rating bias** — 63.9% of reviews are 5-star; the global mean is 4.18; a naive mean predictor achieves RMSE ≈ 1.55
- **Long-tail user activity** — 68.5% of users have exactly one review; collaborative filtering degrades severely for one-time reviewers
- **Noisy labels** — ~15% of reviews are flagged unhelpful by the community (low helpfulness ratio)

The goal: build a fully containerized, end-to-end streaming pipeline that ingests reviews through Kafka, trains a bias-aware ALS model on Spark, serves personalized Top-N recommendations via FastAPI, and exposes a live monitoring dashboard — automatically, from a single `docker compose up`.

---

## Architecture

```
Reviews.csv
    │
    ▼
┌──────────┐   JSON messages (568k)  ┌───────────┐
│ Producer │ ──────────────────────► │   Kafka   │  topic: amazon-reviews
│ (Task 1) │                         │           │
└──────────┘                         └─────┬─────┘
                           ┌───────────────┴────────────────┐
                           │ spark.read (bounded snapshot)   │ spark.readStream (unbounded)
                           ▼                                 ▼
                  ┌─────────────────┐             ┌──────────────────┐
                  │   train.py      │             │   stream.py      │
                  │ ALS + TVS       │             │ frozen model     │
                  │ bias decomp.    │             │ 30s micro-batch  │
                  │ 80/10/10 split  │             │ known users only │
                  └────────┬────────┘             └────────┬─────────┘
                           │                               │
                           └──────────────┬────────────────┘
                                          ▼
                                   ┌────────────┐
                                   │ PostgreSQL │  recommendations · products
                                   │            │  user_biases · item_biases
                                   └─────┬──────┘
                                         │
                                   ┌─────▼──────┐
                                   │  FastAPI   │  REST API · polling feed
                                   └─────┬──────┘
                                         │
                                   ┌─────▼──────┐
                                   │  Dashboard │  Nginx + HTML/JS SPA
                                   └────────────┘

Orchestration: Airflow DAG (recommendation_pipeline · @once · auto-unpaused)
```

**Lambda-inspired split**: `train.py` reads a bounded Kafka snapshot (`spark.read`) for batch ALS training; `stream.py` consumes the same topic as an unbounded stream (`spark.readStream`) for real-time inference using the frozen model. After retraining, Airflow restarts `spark-stream` via Docker socket — the streaming job reloads the new model without interrupting service.

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Message broker | Apache Kafka 3.4 (Confluent 7.5) | Decouples ingestion from training; batch and streaming consumers read the same topic independently |
| Batch processing | Apache Spark 3.5.0 (PySpark) | `spark.read.format("kafka")` for bounded snapshot; MLlib ALS + `TrainValidationSplit` natively distributed |
| Stream processing | Spark Structured Streaming 3.5.0 | `spark.readStream.format("kafka")`; `foreachBatch` handler with PostgreSQL upsert via staging table |
| Orchestration | Apache Airflow 2.9.1 | Enforces task ordering with execution timeouts; LocalExecutor fits single-machine deployment |
| Storage | PostgreSQL 15 | `UNIQUE(user_id, product_id)` enables atomic upsert; stores recommendations, biases, product names |
| API | FastAPI + Uvicorn | Async Python; `psycopg2` connection pool; polling-based live feed endpoint |
| Frontend | Nginx + HTML/JS | Single-page app; Nginx proxies `/api/` to FastAPI; no build step required |
| Containerization | Docker Compose | Reproducible single-command deployment; shared named volumes for model artifacts |

---

## ML Pipeline

### 1. Data preprocessing (`train.py`)

```
Kafka snapshot (568,454 messages)
  → parse JSON: {UserId, ProductId, Score, Time, HelpfulnessNumerator, HelpfulnessDenominator}
  → drop nulls on [UserId, ProductId, Score]
  → filter Score ∈ [1, 5]
  → helpfulness filter: keep h_den = 0 (unvoted) OR h_num/h_den ≥ 0.6
      removes ~15% of reviews the community flagged as unreliable
  → deduplication: one rating per (UserId, ProductId) pair
  → activity filter: MIN_USER_RATINGS = 3, MIN_PRODUCT_RATINGS = 3
  → StringIndexer: UserId → userId (int), ProductId → productId (int)
```

**Result from full 568k dataset**: 235,831 qualifying ratings · 40,456 unique users · 13,494 unique products

### 2. Bias decomposition

Training on raw ratings forces ALS latent factors to model rating scale position rather than genuine preference. Decomposing removes this confound:

```
r(u,i) = μ + b_u + b_i + ε(u,i)

μ     = global mean rating (computed from training split only)
b_u   = per-user bias   = mean(user ratings) − μ
b_i   = per-item bias   = mean(item ratings) − μ
ε     = residual (what ALS actually trains on)
```

**Critical**: biases are computed **after** the 80/10/10 split, from training rows only. Computing biases on the full dataset before splitting constitutes data leakage — it artificially reduces val RMSE to ~0.24 (unrealistic). This leakage was identified and corrected during development.

At inference: `predicted_rating = ε_pred + μ + b_u + b_i`, clamped to [1.0, 5.0].

### 3. ALS training

```
TrainValidationSplit (trainRatio=0.8, parallelism=2)
  Hyperparameter grid:
    rank      ∈ {20, 50, 100}
    regParam  ∈ {0.05, 0.1}
    maxIter   = 15
    → 6 combinations evaluated

Best params: rank=100, regParam=0.05, maxIter=15
coldStartStrategy = "drop", nonnegative = False (residuals can be negative)
```

### 4. Results

| Configuration | Test RMSE | Notes |
|---|---|---|
| Global mean predictor | ~1.55 | Predicts μ = 4.18 for every pair |
| Raw ALS (no bias decomp.) | ~1.03–1.40 | Without residual decomposition |
| **ALS + bias decomposition (200k rows)** | **0.4975** | Earlier smaller run |
| **ALS + bias decomposition (full 568k)** | **0.6764** | Current run — 40,456 users, honest metrics |

Val RMSE: **0.6808** / Test RMSE: **0.6764** — gap of 0.004, confirming no significant overfitting.

The higher RMSE on the full dataset (vs. the 200k run) is expected: the full dataset includes more sparse users (those with exactly 3 reviews), which are harder to model accurately.

---

## Streaming Inference (`stream.py`)

Runs continuously as a Docker service. Loads the frozen ALS model, encoders, and bias tables **once at startup**.

Every 30-second micro-batch:
1. Extract unique `UserId` values from new Kafka messages
2. **Known users** (seen during training): `recommendForUserSubset` → apply bias correction → upsert to PostgreSQL via staging table
3. **Cold-start users** (not in training set): **not written to DB** — the API serves popular-product fallbacks live from `popular_products.json`

This design keeps the recommendations table clean: **398,730 rows, all real ALS predictions, zero cold-start noise**.

---

## Airflow DAG

```
recommendation_pipeline  (@once · auto-unpaused · retries=0)

Task 1: trigger_kafka_ingestion   python3 /producer/producer.py
        MAX_ROWS=568454, DELAY_MS=0 → ~30 min

Task 2: spark_batch_training      spark-submit train.py
        --driver-memory 4g, adaptive query execution → ~20 min

Task 3: restart_streaming_service docker restart spark-stream
        reloads new model, <5 sec

Task 4: print_model_metrics       reads /model/metrics.json, logs RMSE
```

Total pipeline time: **~33 minutes** on a 15 GB laptop.

---

## PostgreSQL Schema

```sql
recommendations  (user_id, product_id, rank, predicted_rating)
                 UNIQUE (user_id, product_id) — enables upsert
user_biases      (user_id, user_bias)         — per-user b_u from training
item_biases      (product_id, item_bias)      — per-item b_i from training
products         (product_id, display_name, review_count) — 74,258 products
model_runs       (val_rmse, test_rmse, best_rank, best_reg, train_rows)
```

All bias values are exposed through the API so the dashboard can render exact score breakdowns: `μ + b_u + b_i + ε = predicted_rating`.

---

## API Reference

```
GET /recommendations/user/{user_id}?n=10
    → personalized Top-N with predicted_rating, user_bias, item_bias, als_residual
    → cold-start users: popular products served live (no DB storage)

GET /users?limit=100
    → sample of personalized users only (ORDER BY RANDOM())

GET /stats
    → personalized_users, cold_start_users, total_recommendation_rows

GET /metrics
    → val_rmse, test_rmse, best_params, global_mean, train_rows, unique_users

GET /metrics/history
    → all training run metrics

GET /health/all
    → component status: postgres, kafka, spark_training, spark_streaming

GET /feed/latest?after=<seq>
    → polling endpoint: returns ≤10 newest Kafka events with seq > after
```

---

## Dashboard

Five-section SPA at `http://localhost:8501`:

| Section | Content |
|---------|---------|
| **Live Feed** | Polls `/feed/latest` every 3 seconds; in-memory ring of 20 events; single `innerHTML` write per poll — no DOM thrashing or browser OOM |
| **Recommendations** | Top-N for any user; ASIN as primary identifier; review summary as quote; exact score breakdown (μ + b_u + b_i + ε) from real bias values |
| **Model Analytics** | RMSE history across runs; hyperparameter table; training data summary |
| **Dataset EDA** | Full-dataset statistics; rating distribution; sparsity analysis; bias motivation |
| **Pipeline Health** | Live component status (postgres, kafka, spark training, spark streaming); 30s auto-refresh |

---

## Deployment

### Prerequisites

- Docker + Docker Compose v2
- 8 GB+ RAM available to Docker (training uses `--driver-memory 4g`)
- `dataset/Reviews.csv` — [Kaggle: Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

### Fresh start

```bash
docker compose down -v && docker compose up --build
```

Pipeline starts automatically. No manual Airflow trigger required. Total time: **~33 minutes**.

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | — |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| FastAPI docs | http://localhost:8000/docs | — |
| Spark cluster UI | http://localhost:8090 | — |
| Spark streaming UI | http://localhost:4040 | — (live after model ready) |
| Spark training UI | http://localhost:4041 | — (active only during Task 2) |
| PostgreSQL | localhost:5433 | `bigdata` / `bigdata` |

### Useful commands

```bash
# Follow training progress
docker compose logs -f airflow

# Manual retrain
docker compose exec airflow airflow dags trigger recommendation_pipeline

# Check recommendation count
docker compose exec postgres psql -U bigdata -d recommendations \
  -c "SELECT COUNT(*), COUNT(DISTINCT user_id) FROM recommendations;"

# Rebuild single service
docker compose up --build api
docker compose up --build dashboard
```

---

## Limitations

**Dataset product names** — `Reviews.csv` contains no official product names; the `Summary` column is reviewer-written text. Display names are derived from the most-helpful review summary per product — descriptive but not authoritative. Fixing this requires an external product catalog (Amazon Product API).

**Single-partition Kafka topic** — 1 partition limits parallel consumption. Spark training and streaming consumers cannot distribute reads across brokers.

**LocalExecutor** — Airflow scheduler and task workers share one container. Concurrent DAG runs or long training jobs can make the webserver unresponsive. Production would use CeleryExecutor or KubernetesExecutor.

**No online model update** — ALS retrains from scratch on each DAG trigger. Incremental updates are not implemented; the model is stale between runs.

**Streaming checkpoint replay** — `startingOffsets="earliest"` with checkpointing means a wiped volume (`down -v`) triggers full Kafka replay on next start.

---

## Future Improvements

- Replace ALS with neural collaborative filtering (NeuMF, LightGCN) for better accuracy on sparse data
- Add content-based features (product category, review text embeddings) to improve cold-start recommendations
- Partition Kafka topic (≥ 3) to enable parallel Spark consumers
- Persist streaming checkpoint to S3/GCS to survive volume wipes
- Add NDCG@K and Precision@K metrics alongside RMSE — RMSE on residuals is not directly interpretable as ranking quality
- Add Prometheus/Grafana for operational monitoring
