# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time product recommendation system using the Amazon Fine Food Reviews dataset (~500k reviews). The pipeline is: Kafka ingestion → Spark ALS training → Airflow orchestration → FastAPI + Nginx dashboard.

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
    train.py       # batch: clean → filter → ALS fit → save model
    stream.py      # streaming: consume Kafka → generate Top-N
airflow/
  dags/            # DAG that chains ingestion → training → streaming
api/               # FastAPI app
dashboard/         # Streamlit app
docker-compose.yml
```

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
1. Start Kafka producer (or trigger it via BashOperator/DockerOperator)
2. Run Spark batch training job
3. Run Spark streaming job (or keep it always-on as a separate service)

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

Submit batch training:
```bash
docker compose exec spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:<version> \
  /jobs/train.py
```

Submit streaming job:
```bash
docker compose exec spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:<version> \
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
- Dashboard: `http://localhost:8501`
