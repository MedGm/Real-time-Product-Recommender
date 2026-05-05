# Real-Time Product Recommender — Big Data Pipeline

> Projet du module Big Data · Amazon Fine Food Reviews (~568k reviews) · ALS Collaborative Filtering

---

## Architecture

![Architecture diagram](architecture.png)

Lambda-inspired pipeline — Kafka ingestion → Spark ALS batch training → Spark Structured Streaming inference → FastAPI → Dashboard.

---

## Quick Start

### Prerequisites
- Docker + Docker Compose v2
- ≥ 8 GB RAM available to Docker
- `dataset/Reviews.csv` present (download from [Kaggle: amazon-fine-food-reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews))

### Launch (fully automatic)

```bash
docker compose down -v   # clean start — wipes old volumes
docker compose up --build
```

The pipeline is **fully automated**. Once all services are healthy:
1. Airflow's `recommendation_pipeline` DAG auto-triggers (no manual click needed)
2. Task 1 streams 200,000 reviews into Kafka
3. Task 2 trains the ALS model with bias decomposition
4. Task 3 reloads the streaming inference service
5. Task 4 logs RMSE to Airflow

Total time from cold start to recommendations available: **~25–35 minutes**.

### Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| **Dashboard** | http://localhost:8501 | — |
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **FastAPI docs** | http://localhost:8000/docs | — |
| **Spark UI** | http://localhost:8090 | — |

---

## Pipeline Components

### Kafka Ingestion

The Kafka producer (`producer/producer.py`) streams `Reviews.csv` row-by-row as JSON messages to the `amazon-reviews` topic. Each message carries:

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

The producer runs as **Airflow DAG Task 1** (not as a standalone service), ensuring the training job always reads a complete, consistent snapshot.

### Spark ALS Batch Training (`spark/jobs/train.py`)

**Data cleaning:**
- Drop null users, products, or ratings
- Filter ratings outside [1, 5]
- **Helpfulness filter**: keep only reviews with no community votes OR helpfulness ratio ≥ 60 % — removes ~15 % of noisy reviews
- Deduplicate on `(UserId, ProductId)` — one rating per user-product pair
- Activity filter: drop users/products with fewer than 3 ratings

**Bias decomposition (key innovation):**

ALS on Amazon reviews suffers from extreme positive bias (63.9 % of ratings are 5-star, median = 5). The solution is to decompose each rating before training:

```
r_ui = μ + b_u + b_i + ε_ui
```

where μ = global mean (4.18), b_u = user bias, b_i = item bias, ε = residual.

ALS trains on the **residual** ε only. At inference, predictions are reconstructed as `ε_pred + μ + b_u + b_i`. This frees latent factors from modeling bias and captures genuine collaborative-filtering signal.

**Hyperparameter tuning:**
- Spark MLlib `TrainValidationSplit` (not a hand-rolled loop)
- Grid: `rank ∈ {20, 50, 100}`, `regParam ∈ {0.05, 0.1}`, `maxIter = 15` → 6 combinations
- 80/10/10 train/validation/test split (test set never seen during tuning)
- Metric: RMSE

**Achieved performance:**

| Configuration | RMSE |
|---|---|
| Baseline (global mean predictor) | ~1.5–1.7 |
| ALS raw ratings, rank∈{10,20}, reg∈{0.01,0.1} | ~1.3–1.4 |
| ALS + bias decomposition + expanded grid | **0.557 (test)** |

### Spark Structured Streaming (`spark/jobs/stream.py`)

Always-on service consuming the Kafka topic with 30-second micro-batches. For each batch:
- **Known users**: `recommendForUserSubset` → apply bias correction → upsert Top-N to PostgreSQL
- **Cold-start users**: popular-product fallback (Top-N most-reviewed products from training)

Bias correction applied: `predicted_rating = ALS_residual + μ + b_u + b_i`

### Airflow Orchestration (`airflow/dags/recommendation_pipeline.py`)

Fully automated DAG (`schedule='@once'`, auto-unpaused):

```
trigger_kafka_ingestion → spark_batch_training → restart_streaming_service → print_model_metrics
```

| Task | What it does |
|------|-------------|
| `trigger_kafka_ingestion` | Runs `producer.py` directly inside Airflow container (200k messages, no delay) |
| `spark_batch_training` | `spark-submit train.py` — full pipeline: clean → bias decompose → TVS → save |
| `restart_streaming_service` | `docker restart spark-stream` — streaming job reloads new model |
| `print_model_metrics` | Reads `/model/metrics.json`, logs RMSE + best params to Airflow |

To retrain manually (e.g. after more data accumulates):
```bash
docker compose exec airflow airflow dags trigger recommendation_pipeline
```

### FastAPI (`api/main.py`)

Connection-pooled REST API serving recommendations from PostgreSQL.

```bash
# Get Top-N recommendations
GET /recommendations/user/{user_id}?n=10
→ {"user_id": "A3SGXH7AUHU8GW", "recommendations": ["B001E4KFG0", ...], "predicted_ratings": [4.87, ...]}

# Model metrics
GET /metrics
→ {"val_rmse": 0.566, "test_rmse": 0.557, "best_params": {"rank": 100, "regParam": 0.05, "maxIter": 15}, "global_mean": 4.18, ...}

# Pipeline status
GET /pipeline-status
→ {"model_ready": true, "unique_users": 11751, "unique_products": 3033, ...}

# Aggregate stats
GET /stats
→ {"users_with_recommendations": 11751, "total_recommendation_rows": 117510}
```

### Dashboard (`dashboard/index.html`)

Static single-page app served by Nginx. Nginx reverse-proxies `/api/` → FastAPI. No Python runtime needed.

Pages:
- **Overview** — pipeline status, live stats (users indexed, recommendations stored, RMSE)
- **Recommendations** — look up Top-N for any user, click indexed user chips
- **Model Analytics** — RMSE comparison chart, data split donut, hyperparameter table, training run info
- **Pipeline Status** — model readiness, service port reference

### PostgreSQL (`postgres/`)

Two tables:
- `recommendations` — `UNIQUE(user_id, product_id)`, upserted via staging table pattern
- `model_runs` — history of each training run (RMSE, best params, row count)

---

## Data Splits

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Training | **80%** | ALS residual matrix factorization |
| Validation | **10%** | `TrainValidationSplit` hyperparameter tuning |
| Test (held-out) | **10%** | Final RMSE — model never sees this during training |

---

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ROWS` | `200000` | Messages streamed to Kafka per pipeline run |
| `MIN_USER_RATINGS` | `3` | Minimum ratings for a user to enter training |
| `MIN_PRODUCT_RATINGS` | `3` | Minimum ratings for a product to enter training |
| `TOP_N` | `10` | Recommendations generated per user |
| `DELAY_MS` | `0` (DAG) / `5` (manual) | ms between Kafka messages |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| DAG does not auto-start | Check Airflow scheduler: `docker compose logs airflow \| grep scheduler`. DAG should be unpaused automatically. |
| Training fails with too few rows | `docker compose logs airflow \| grep "Rows after filtering"`. If < 100, volumes may be stale — run `down -v` first. |
| `spark_batch_training` OOM | Increase Docker memory limit to ≥ 8 GB. |
| Dashboard shows "API offline" | Wait ~2 min for full stack startup; check `docker compose logs api`. |
| Streaming not generating recs | `docker exec spark-stream ls /model/` — verify `metrics.json`, `user_biases/`, `item_biases/` exist. |
| Port conflict on 8080/8501 | Stop conflicting processes or change ports in `docker-compose.yml`. |

---

## Teardown

```bash
# Full teardown including all data volumes
docker compose down -v

# Rebuild single service without wiping data
docker compose up --build api
docker compose up --build dashboard
```
