# Big Data — Real-time Product Recommender

Amazon Fine Food Reviews → Kafka → Spark ALS → FastAPI + Streamlit

## Architecture

```
dataset/Reviews.csv
        │
        ▼
[producer] ──► Kafka (amazon-reviews topic)
                │
                ▼
          [spark train.py]  ── ALS model ──► /model/als
          [spark stream.py] ◄── Kafka ──► PostgreSQL (recommendations)
                │
          [Airflow DAG]  orchestrates the above
                │
          [FastAPI :8000]  GET /recommendations/user/{id}
          [Streamlit :8501] dashboard
```

## Requirements

- Docker + Docker Compose v2
- ~4 GB RAM for Spark (2 GB worker)
- `dataset/Reviews.csv` present

## Quick start

```bash
# 1. (optional) adjust speed / subset size
cp .env .env.local   # edit DELAY_MS, MAX_ROWS

# 2. Launch everything
docker compose up --build

# 3. Airflow UI  → http://localhost:8080  (airflow / airflow)
#    Spark UI    → http://localhost:8090
#    API docs    → http://localhost:8000/docs
#    Dashboard   → http://localhost:8501
```

## Pipeline steps

| Step | What happens |
|------|-------------|
| `kafka-init` | Creates the `amazon-reviews` topic |
| `producer` | Streams Reviews.csv rows as JSON to Kafka |
| Airflow DAG `recommendation_pipeline` | Waits for 1 000 messages → triggers Spark training → starts streaming job |
| `spark train.py` | Cleans data, trains ALS (80/10/10 split), saves model + metrics |
| `spark stream.py` | Consumes Kafka in micro-batches, generates Top-N, writes to Postgres |
| FastAPI | Serves `/recommendations/user/{id}` from Postgres |
| Streamlit | Interactive dashboard over the API |

## Triggering the DAG manually

```bash
docker compose exec airflow airflow dags trigger recommendation_pipeline
```

## Running Spark jobs directly (bypass Airflow)

```bash
# Training
docker compose exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /jobs/train.py

# Streaming
docker compose exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.3 \
  /jobs/stream.py
```

## Data splits

- **80%** training
- **10%** validation / hyperparameter grid search (rank, regParam)
- **10%** held-out test — RMSE reported in `/model/metrics.json` and `GET /metrics`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DELAY_MS` | `10` | Sleep between Kafka messages (ms) |
| `MAX_ROWS` | `50000` | Max rows from CSV (0 = all) |
| `TOP_N` | `10` | Recommendations per user |
| `MIN_USER_RATINGS` | `5` | Min reviews to keep a user |
| `MIN_PRODUCT_RATINGS` | `5` | Min reviews to keep a product |
