"""
FastAPI — recommendations service.

Endpoints:
  GET /recommendations/user/{user_id}?n=10
  GET /users                              — list of known user IDs
  GET /metrics                            — last model RMSE
  GET /health
"""

import json
import os
from typing import List, Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_URL     = os.getenv("DB_URL", "postgresql://bigdata:bigdata@postgres/recommendations")
MODEL_PATH = os.getenv("MODEL_PATH", "/model")

app = FastAPI(title="Recommender API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    return psycopg2.connect(DB_URL)


# ── Models ────────────────────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]

class MetricsResponse(BaseModel):
    val_rmse: float
    test_rmse: float
    best_params: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=50),
):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT product_id
                FROM recommendations
                WHERE user_id = %s
                ORDER BY rank
                LIMIT %s
                """,
                (user_id, n),
            )
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user '{user_id}'. "
                   "The model may not have seen this user yet.",
        )

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[r["product_id"] for r in rows],
    )


@app.get("/users")
def list_users(limit: int = Query(default=100, ge=1, le=1000)):
    """Return a sample of user IDs that have recommendations."""
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
        raise HTTPException(status_code=404, detail="Model metrics not available yet.")


@app.get("/stats")
def get_stats():
    """High-level stats: total users with recommendations, total rows."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT user_id), COUNT(*) FROM recommendations")
            users, total = cur.fetchone()
    return {"users_with_recommendations": users, "total_recommendation_rows": total}
