"""
FastAPI — recommendations service.

Endpoints:
  GET /health
  GET /pipeline-status                        — model training status
  GET /recommendations/user/{user_id}?n=10    — Top-N product IDs
  GET /users?limit=100                        — sample of indexed users
  GET /metrics                                — last model RMSE + params
  GET /stats                                  — aggregate counts
"""

import json
import os
import time
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

# ── Connection pool ───────────────────────────────────────────────────────────

_pool: Optional[pg_pool.SimpleConnectionPool] = None

def get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        for attempt in range(10):
            try:
                _pool = pg_pool.SimpleConnectionPool(1, 20, DB_URL)
                break
            except psycopg2.OperationalError as e:
                print(f"Waiting for postgres... attempt {attempt + 1}/10")
                time.sleep(3)
        else:
            raise RuntimeError("Could not connect to postgres after 10 attempts")
    return _pool


@contextmanager
def get_conn():
    """Context manager that borrows a connection from the pool and returns it."""
    conn = get_pool().getconn()
    try:
        yield conn
    finally:
        get_pool().putconn(conn)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the connection pool on startup
    get_pool()
    yield
    # Close all connections on shutdown
    if _pool:
        _pool.closeall()


app = FastAPI(
    title       = "Real-Time Product Recommender API",
    description = "Amazon Fine Food Reviews · Spark ALS · Big Data Pipeline",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    user_id:         str
    recommendations: List[str]
    predicted_ratings: Optional[List[float]] = None


class MetricsResponse(BaseModel):
    val_rmse:        float
    test_rmse:       float
    best_params:     dict
    trained_at:      Optional[str] = None
    finished_at:     Optional[str] = None
    train_rows:      Optional[int] = None
    unique_users:    Optional[int] = None
    unique_products: Optional[int] = None


class PipelineStatus(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_ready:    bool
    trained_at:     Optional[str] = None
    finished_at:    Optional[str] = None
    val_rmse:       Optional[float] = None
    test_rmse:      Optional[float] = None
    train_rows:     Optional[int] = None
    unique_users:   Optional[int] = None
    unique_products: Optional[int] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pipeline-status", response_model=PipelineStatus)
def pipeline_status():
    """Reports whether the ALS model has been trained and is ready."""
    metrics_file = f"{MODEL_PATH}/metrics.json"
    if not os.path.exists(metrics_file):
        return PipelineStatus(
            model_ready    = False,
            trained_at     = None,
            finished_at    = None,
            val_rmse       = None,
            test_rmse      = None,
            train_rows     = None,
            unique_users   = None,
            unique_products = None,
        )
    with open(metrics_file) as f:
        m = json.load(f)
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


@app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=50),
):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT product_id, predicted_rating
                FROM   recommendations
                WHERE  user_id = %s
                ORDER  BY rank
                LIMIT  %s
                """,
                (user_id, n),
            )
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code = 404,
            detail      = (
                f"No recommendations found for user '{user_id}'. "
                "The streaming job may not have processed this user yet — "
                "ensure the Airflow DAG has completed training."
            ),
        )

    return RecommendationResponse(
        user_id           = user_id,
        recommendations   = [r["product_id"] for r in rows],
        predicted_ratings = [round(r["predicted_rating"], 3) if r["predicted_rating"] else None for r in rows],
    )


@app.get("/users")
def list_users(limit: int = Query(default=100, ge=1, le=1000)):
    """Return a sample of user IDs that have recommendations stored."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT user_id FROM recommendations LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    return {"users": [r[0] for r in rows]}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    metrics_file = f"{MODEL_PATH}/metrics.json"
    try:
        with open(metrics_file) as f:
            m = json.load(f)
        return MetricsResponse(**m)
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = "Model metrics not available yet. Trigger the Airflow DAG first.",
        )


@app.get("/stats")
def get_stats():
    """High-level stats: unique users indexed and total recommendation rows."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT user_id), COUNT(*) FROM recommendations")
            users, total = cur.fetchone()
    return {
        "users_with_recommendations":  users,
        "total_recommendation_rows":   total,
    }
