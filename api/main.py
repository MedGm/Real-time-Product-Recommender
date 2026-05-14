"""
FastAPI — recommendations service.

Endpoints:
  GET  /health                              — liveness probe
  GET  /health/all                          — detailed component health
  GET  /pipeline-status                     — model training status
  GET  /recommendations/user/{user_id}?n=10 — Top-N product IDs + names
  GET  /users?limit=100                     — sample of indexed users
  GET  /users/{user_id}/profile             — user bias, avg rating, rec count
  GET  /metrics                             — last model RMSE + params
  GET  /metrics/history                     — all training runs
  GET  /stats                               — aggregate counts
  GET  /products/{product_id}               — product display name
  GET  /dataset/ratings                     — rating distribution for EDA panel
  GET  /feed/latest?after=<ts_ms>           — poll ring buffer (replaces SSE)
"""

import asyncio
import glob
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from typing import List, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_URL     = os.getenv("DB_URL", "postgresql://bigdata:bigdata@postgres/recommendations")
MODEL_PATH = os.getenv("MODEL_PATH", "/model")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "amazon-reviews")

# ── Connection pool ───────────────────────────────────────────────────────────

_pool: Optional[pg_pool.SimpleConnectionPool] = None


def get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        for attempt in range(10):
            try:
                _pool = pg_pool.SimpleConnectionPool(1, 20, DB_URL)
                break
            except psycopg2.OperationalError:
                print(f"Waiting for postgres... attempt {attempt + 1}/10")
                time.sleep(3)
        else:
            raise RuntimeError("Could not connect to postgres after 10 attempts")
    return _pool


@contextmanager
def get_conn():
    conn = get_pool().getconn()
    try:
        yield conn
    finally:
        get_pool().putconn(conn)


# ── Feed ring buffer ──────────────────────────────────────────────────────────

_feed_buffer: deque = deque(maxlen=200)   # last 200 Kafka events, in memory only
_feed_seq:    int   = 0                   # monotonic counter — never resets


async def _kafka_feed_task():
    """Background task: consume Kafka topic, fill ring buffer. Retries on error."""
    global _feed_seq
    while True:
        consumer = None
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers  = KAFKA_BOOTSTRAP,
                auto_offset_reset  = "latest",
                enable_auto_commit = False,
                value_deserializer = lambda b: json.loads(b.decode("utf-8")),
            )
            loop = asyncio.get_event_loop()
            while True:
                records = await loop.run_in_executor(
                    None, lambda: consumer.poll(timeout_ms=1000, max_records=10)
                )
                for _tp, msgs in records.items():
                    for msg in msgs:
                        _feed_seq += 1
                        _feed_buffer.append({
                            "seq":        _feed_seq,          # unique — never collides
                            "user_id":    msg.value.get("UserId"),
                            "product_id": msg.value.get("ProductId"),
                            "score":      msg.value.get("Score"),
                            "summary":    msg.value.get("Summary", ""),
                            "ts":         msg.timestamp,
                        })
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[feed] Kafka consumer error: {e} — retrying in 5s")
            await asyncio.sleep(5)
        finally:
            if consumer:
                try:
                    consumer.close()
                except Exception:
                    pass


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pool()
    task = asyncio.create_task(_kafka_feed_task())
    yield
    task.cancel()
    if _pool:
        _pool.closeall()


app = FastAPI(
    title       = "Real-Time Product Recommender API",
    description = "Amazon Fine Food Reviews · Spark ALS · Big Data Pipeline",
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    rank:             int
    product_id:       str
    display_name:     Optional[str] = None
    predicted_rating: Optional[float] = None
    item_bias:        Optional[float] = None
    als_residual:     Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id:         str
    is_cold_start:   bool
    user_bias:       Optional[float] = None
    recommendations: List[RecommendationItem]


class MetricsResponse(BaseModel):
    val_rmse:        float
    test_rmse:       float
    best_params:     dict
    global_mean:     Optional[float] = None
    trained_at:      Optional[str] = None
    finished_at:     Optional[str] = None
    train_rows:      Optional[int] = None
    unique_users:    Optional[int] = None
    unique_products: Optional[int] = None


class PipelineStatus(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_ready:     bool
    trained_at:      Optional[str] = None
    finished_at:     Optional[str] = None
    val_rmse:        Optional[float] = None
    test_rmse:       Optional[float] = None
    train_rows:      Optional[int] = None
    unique_users:    Optional[int] = None
    unique_products: Optional[int] = None


class UserProfile(BaseModel):
    user_id:       str
    is_known:      bool
    rec_count:     int
    avg_predicted: Optional[float] = None


class ComponentHealth(BaseModel):
    name:   str
    status: str
    detail: str


class HealthAll(BaseModel):
    overall:    str
    components: List[ComponentHealth]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_metrics() -> Optional[dict]:
    try:
        with open(f"{MODEL_PATH}/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _load_popular_products() -> List[str]:
    try:
        with open(f"{MODEL_PATH}/popular_products.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _lookup_product_names(product_ids: List[str]) -> dict:
    if not product_ids:
        return {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT product_id, display_name FROM products WHERE product_id = ANY(%s)",
                (product_ids,),
            )
            return {r[0]: r[1] for r in cur.fetchall()}


# ──────────────────────────────────────────────────────────────────────────────
# ── API ENDPOINTS ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# ── SECTION 1: System Monitoring ──────────────────────────────────────────────

@app.get("/health")
def health():
    """Basic liveness probe for load balancer / orchestration."""
    return {"status": "ok"}


@app.get("/health/all", response_model=HealthAll)
def health_all():
    """Detailed health check of all big data components (Postgres, Kafka, Spark)."""
    components: List[ComponentHealth] = []

    # Postgres connectivity
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM recommendations")
                rows = cur.fetchone()[0]
        components.append(ComponentHealth(
            name="postgres", status="healthy",
            detail=f"{rows:,} recommendation rows",
        ))
    except Exception as e:
        components.append(ComponentHealth(name="postgres", status="offline", detail=str(e)))

    # Kafka topic status
    try:
        from kafka import KafkaConsumer
        from kafka.structs import TopicPartition
        consumer   = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP)
        partitions = consumer.partitions_for_topic(KAFKA_TOPIC) or set()
        tps        = [TopicPartition(KAFKA_TOPIC, p) for p in partitions]
        end_offs   = consumer.end_offsets(tps) if tps else {}
        begin_offs = consumer.beginning_offsets(tps) if tps else {}
        total      = sum(end_offs[tp] - begin_offs[tp] for tp in tps)
        consumer.close()
        components.append(ComponentHealth(
            name="kafka", status="healthy",
            detail=f"{total:,} messages in {KAFKA_TOPIC}",
        ))
    except Exception as e:
        components.append(ComponentHealth(name="kafka", status="offline", detail=str(e)))

    # Batch Training Metrics
    m = _load_metrics()
    if m:
        components.append(ComponentHealth(
            name="spark_training", status="healthy",
            detail=f"test RMSE {m.get('test_rmse', 0):.4f} · finished {m.get('finished_at', '?')[:10]}",
        ))
    else:
        components.append(ComponentHealth(
            name="spark_training", status="degraded",
            detail="metrics.json not found — training not yet complete",
        ))

    # Streaming Checkpoint Recency
    checkpoint = f"{MODEL_PATH}/stream_checkpoint"
    try:
        files = glob.glob(f"{checkpoint}/**/*", recursive=True)
        if files:
            latest = max(os.path.getmtime(f) for f in files if os.path.isfile(f))
            age_s  = int(time.time() - latest)
            status = "healthy" if age_s < 120 else "degraded"
            components.append(ComponentHealth(
                name="spark_streaming", status=status,
                detail=f"last checkpoint {age_s}s ago",
            ))
        else:
            components.append(ComponentHealth(
                name="spark_streaming", status="degraded",
                detail="no checkpoint yet",
            ))
    except Exception as e:
        components.append(ComponentHealth(name="spark_streaming", status="offline", detail=str(e)))

    overall = (
        "healthy" if all(c.status == "healthy" for c in components) else
        "offline" if any(c.status == "offline"  for c in components) else
        "degraded"
    )
    return HealthAll(overall=overall, components=components)


@app.get("/pipeline-status", response_model=PipelineStatus)
def pipeline_status():
    """Returns the high-level readiness of the machine learning pipeline."""
    m = _load_metrics()
    if not m:
        return PipelineStatus(model_ready=False)
    return PipelineStatus(
        model_ready     = True,
        trained_at      = m.get("trained_at"),
        finished_at     = m.get("finished_at"),
        val_rmse        = m.get("val_rmse"),
        test_rmse       = m.get("test_rmse"),
        train_rows      = m.get("train_rows"),
        unique_users    = m.get("unique_users"),
        unique_products = m.get("unique_products"),
    )


# ── SECTION 2: User Intelligence ──────────────────────────────────────────────

@app.get("/users")
def list_users(limit: int = Query(default=100, ge=1, le=1000)):
    """Returns a random sample of indexed users with pre-calculated recommendations."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id FROM recommendations
                WHERE predicted_rating IS NOT NULL
                GROUP BY user_id
                ORDER BY RANDOM()
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return {"users": [r[0] for r in rows]}


@app.get("/users/{user_id}/profile", response_model=UserProfile)
def user_profile(user_id: str):
    """Retrieves metadata and aggregate stats for a specific user profile."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), AVG(predicted_rating) FROM recommendations WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()

    if not row or row[0] == 0:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    rec_count, avg_pred = row
    return UserProfile(
        user_id       = user_id,
        is_known      = avg_pred is not None,
        rec_count     = rec_count,
        avg_predicted = round(avg_pred, 3) if avg_pred else None,
    )


# ── SECTION 3: Inference Engine ───────────────────────────────────────────────

@app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=50),
):
    """
    Main entry point for serving Top-N recommendations.
    Includes bias-decomposition metadata for explainable AI.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT r.product_id, r.predicted_rating, r.rank,
                       ib.item_bias
                FROM   recommendations r
                LEFT JOIN item_biases ib ON ib.product_id = r.product_id
                WHERE  r.user_id = %s
                ORDER  BY r.rank
                LIMIT  %s
                """,
                (user_id, n),
            )
            rows = cur.fetchall()

            cur.execute(
                "SELECT user_bias FROM user_biases WHERE user_id = %s",
                (user_id,),
            )
            ub_row = cur.fetchone()

    if not rows:
        # Unknown user — serve popular products live, no DB write needed
        popular     = _load_popular_products()[:n]
        name_map    = _lookup_product_names(popular)
        items       = [
            RecommendationItem(
                rank             = i + 1,
                product_id       = pid,
                display_name     = name_map.get(pid),
                predicted_rating = None,
            )
            for i, pid in enumerate(popular)
        ]
        return RecommendationResponse(
            user_id         = user_id,
            is_cold_start   = True,
            user_bias       = None,
            recommendations = items,
        )

    is_cold_start = all(r["predicted_rating"] is None for r in rows)
    product_ids   = [r["product_id"] for r in rows]
    name_map      = _lookup_product_names(product_ids)
    user_bias     = round(ub_row["user_bias"], 4) if ub_row else None
    global_mean   = _load_metrics().get("global_mean", 0.0) if _load_metrics() else 0.0

    items = []
    for r in rows:
        pr = round(min(5.0, max(1.0, r["predicted_rating"])), 3) if r["predicted_rating"] else None
        ib = round(r["item_bias"], 4) if r["item_bias"] is not None else None
        als = None
        if pr is not None and user_bias is not None and ib is not None:
            als = round(pr - global_mean - user_bias - ib, 4)
        items.append(RecommendationItem(
            rank             = r["rank"],
            product_id       = r["product_id"],
            display_name     = name_map.get(r["product_id"]),
            predicted_rating = pr,
            item_bias        = ib,
            als_residual     = als,
        ))

    return RecommendationResponse(
        user_id         = user_id,
        is_cold_start   = is_cold_start,
        user_bias       = user_bias,
        recommendations = items,
    )


@app.get("/products/{product_id}")
def get_product(product_id: str):
    """Fetches a specific product's display name and overall review volume."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT display_name, review_count FROM products WHERE product_id = %s",
                (product_id,),
            )
            row = cur.fetchone()
    if not row:
        return {"product_id": product_id, "display_name": None, "review_count": None}
    return {"product_id": product_id, "display_name": row[0], "review_count": row[1]}


# ── SECTION 4: Model Analytics ────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(timestamp: Optional[str] = None):
    """Retrieves RMSE and hyperparameter metadata for the current or a specific model run."""
    if timestamp:
        history_file = f"{MODEL_PATH}/metrics_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
                    # Find the run with the matching finished_at timestamp
                    for run in history:
                        if run.get("finished_at") == timestamp:
                            valid = {k: v for k, v in run.items() if k in MetricsResponse.model_fields}
                            return MetricsResponse(**valid)
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Metrics for timestamp {timestamp} not found.")

    m = _load_metrics()
    if not m:
        raise HTTPException(status_code=404, detail="Model metrics not available yet.")
    valid = {k: v for k, v in m.items() if k in MetricsResponse.model_fields}
    return MetricsResponse(**valid)


@app.get("/metrics/history")
def get_metrics_history():
    """Returns the full history of all successful batch training runs."""
    history_file = f"{MODEL_PATH}/metrics_history.json"
    if not os.path.exists(history_file):
        # Fallback: if no history yet, return current metrics as a single-item list
        m = _load_metrics()
        return [m] if m else []
    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/stats")
def get_stats():
    """Aggregates high-level counts for the recommendations database."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT CASE WHEN predicted_rating IS NOT NULL THEN user_id END),
                    COUNT(DISTINCT CASE WHEN predicted_rating IS NULL     THEN user_id END),
                    COUNT(DISTINCT user_id),
                    COUNT(*)
                FROM recommendations
            """)
            personalized, cold_start, total_users, total_recs = cur.fetchone()
    return {
        "users_with_recommendations": total_users,
        "personalized_users":         personalized,
        "cold_start_users":           cold_start,
        "total_recommendation_rows":  total_recs,
    }


@app.get("/dataset/ratings")
def dataset_ratings():
    """Static EDA stats for the Amazon Fine Food Reviews dataset."""
    return {
        "distribution": [
            {"stars": 1, "count": 52268,  "pct": 9.2},
            {"stars": 2, "count": 29769,  "pct": 5.2},
            {"stars": 3, "count": 42640,  "pct": 7.5},
            {"stars": 4, "count": 80655,  "pct": 14.2},
            {"stars": 5, "count": 363122, "pct": 63.9},
        ],
        "total":            568454,
        "global_mean":      4.18,
        "median":           5.0,
        "sparsity":         99.9971,
        "unique_users":     256059,
        "unique_products":  74258,
    }


# ── SECTION 5: Real-time Streams ──────────────────────────────────────────────

@app.get("/feed/latest")
def feed_latest(after: int = Query(default=0)):
    """
    Returns a window of the newest Kafka events. 
    Uses sequence-based polling for reliability.
    """
    events = [e for e in _feed_buffer if e["seq"] > after]
    return {"events": events[-10:]}

