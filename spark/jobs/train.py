"""
Spark batch job — trains an ALS model on Amazon Fine Food Reviews.

Data source: Kafka topic `amazon-reviews` (bounded snapshot via spark.read).
Kafka acts as the streaming ingestion layer; this job reads a static snapshot
of all accumulated messages using spark.read.format("kafka") with
endingOffsets="latest". This materialises the interactions into a DataFrame
before ALS training begins — ALS remains a pure batch algorithm.

  spark.read  → bounded read (stops at current end of topic) → used HERE for training
  spark.readStream → unbounded read (runs forever)           → used by stream.py for inference

Splits:
  80% train | 10% validation (hyperparameter tuning) | 10% test (held-out)

Outputs written to MODEL_PATH (shared Docker volume):
  /model/als                   — saved ALS PipelineModel
  /model/encoders              — saved StringIndexer models (user + product)
  /model/test_data             — parquet of the 10% held-out set (for offline eval)
  /model/metrics.json          — val RMSE, test RMSE, best params, timestamps
  /model/popular_products.json — top-N most-rated products (cold-start fallback for stream.py)

Also writes Top-N recommendations for ALL known users directly to PostgreSQL
at training time (Option B): this ensures the API can serve recs immediately
after training without waiting for stream.py to process each user individually.
stream.py then handles only new/unknown users via popular-product fallbacks.

Architecture note (Lambda-inspired split):
  Kafka provides the streaming ingestion layer. The producer streams Reviews.csv
  row-by-row to the topic. The Airflow sensor waits until KAFKA_MIN_MSGS messages
  have accumulated, then triggers this job. This job reads the full snapshot,
  trains ALS, writes recommendations + popular products, and saves the model.
  The spark-stream service (always-on) is then restarted by Airflow to handle
  cold-start users via popular-product fallbacks.
"""

import json
import os
import shutil
from datetime import datetime, timezone

import psycopg2

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, from_json, posexplode
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

# ── Kafka config ──────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "amazon-reviews")

# ── Output path ───────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/model")

# ── PostgreSQL config ─────────────────────────────────────────────────────────
_PG_HOST = os.getenv("POSTGRES_HOST",     "postgres")
_PG_PORT = os.getenv("POSTGRES_PORT",     "5432")
_PG_DB   = os.getenv("POSTGRES_DB",       "recommendations")
_PG_USER = os.getenv("POSTGRES_USER",     "bigdata")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "bigdata")

# ── Filtering thresholds — drop rare users / items ────────────────────────────
MIN_USER_RATINGS    = int(os.getenv("MIN_USER_RATINGS",    "5"))
MIN_PRODUCT_RATINGS = int(os.getenv("MIN_PRODUCT_RATINGS", "5"))

# ── Top-N recommendations per user ────────────────────────────────────────────
TOP_N = int(os.getenv("TOP_N", "10"))

# ── JDBC config for writing recommendations to PostgreSQL ─────────────────────
JDBC_URL  = f"jdbc:postgresql://{os.getenv('POSTGRES_HOST','postgres')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','recommendations')}"
JDBC_PROP = {
    "user":     os.getenv("POSTGRES_USER",     "bigdata"),
    "password": os.getenv("POSTGRES_PASSWORD", "bigdata"),
    "driver":   "org.postgresql.Driver",
}

# ── Kafka message schema ──────────────────────────────────────────────────────
EVENT_SCHEMA = StructType([
    StructField("UserId",    StringType(), True),
    StructField("ProductId", StringType(), True),
    StructField("Score",     FloatType(),  True),
    StructField("Time",      LongType(),   True),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ALS-Training")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def load_from_kafka(spark: SparkSession):
    """
    Read a bounded snapshot from the Kafka topic.

    Uses spark.read (not spark.readStream) so the result is a static DataFrame
    that ALS can train on. endingOffsets="latest" ensures the read stops at the
    current end of the topic — new messages produced after this point are not
    included in this training run.
    """
    print(f"[train] Reading bounded Kafka snapshot from topic '{KAFKA_TOPIC}' ...")

    raw = (
        spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("endingOffsets",   "latest")     # bounded — stops here
        .option("failOnDataLoss",  "false")
        .load()
    )

    df = (
        raw
        .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("d"))
        .select(
            col("d.UserId").alias("raw_user"),
            col("d.ProductId").alias("raw_product"),
            col("d.Score").cast("float").alias("rating"),
        )
        .dropna()
        .filter((col("rating") >= 1) & (col("rating") <= 5))
        # Guard against producer re-runs writing the same events twice into the topic.
        # Without deduplication, training on a replayed topic inflates rating counts
        # and biases the ALS matrix toward power users.
        .dropDuplicates(["raw_user", "raw_product", "rating"])
    )

    return df


def filter_active(df):
    """Drop users/items with fewer than the minimum number of ratings."""

    # Filter active users
    user_counts  = df.groupBy("raw_user").agg(count("*").alias("cnt"))
    active_users = user_counts.filter(col("cnt") >= MIN_USER_RATINGS).select("raw_user")
    df = df.join(active_users, on="raw_user", how="inner")

    # Filter active products
    prod_counts  = df.groupBy("raw_product").agg(count("*").alias("cnt"))
    active_prods = prod_counts.filter(col("cnt") >= MIN_PRODUCT_RATINGS).select("raw_product")
    df = df.join(active_prods, on="raw_product", how="inner")

    row_count = df.count()
    print(f"[train] Rows after filtering: {row_count:,}")
    return df, row_count


def encode(df):
    """Encode string user/product IDs to sequential integers for ALS."""
    user_indexer    = StringIndexer(inputCol="raw_user",    outputCol="userId",    handleInvalid="skip")
    product_indexer = StringIndexer(inputCol="raw_product", outputCol="productId", handleInvalid="skip")
    pipeline        = Pipeline(stages=[user_indexer, product_indexer])
    encoder         = pipeline.fit(df)
    encoded         = encoder.transform(df).select(
        col("userId").cast("int"),
        col("productId").cast("int"),
        col("rating"),
        col("raw_user"),
        col("raw_product"),
    )
    return encoded, encoder


def compute_popular_products(df) -> list:
    """
    Compute the TOP_N most-rated products from the filtered training data.

    Saved to /model/popular_products.json as a cold-start fallback:
    stream.py inserts these products for users not seen during ALS training.
    """
    popular = (
        df
        .groupBy("raw_product")
        .agg(count("*").alias("cnt"))
        .orderBy(desc("cnt"))
        .limit(TOP_N)
        .select("raw_product")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    path = f"{MODEL_PATH}/popular_products.json"
    with open(path, "w") as f:
        json.dump(popular, f, indent=2)
    print(f"[train] Saved {len(popular)} popular products to {path}.")
    return popular


def write_all_recommendations(spark: SparkSession, best_model, encoder):
    """
    Generate Top-N recommendations for ALL known users using the trained ALS model
    and upsert them to PostgreSQL (Option B from architecture review).

    This guarantees the API can serve recommendations immediately after training
    without waiting for stream.py to process each user via Kafka micro-batches.
    stream.py then only needs to handle cold-start users (new/unknown).

    Uses a staging table + ON CONFLICT DO UPDATE for safe idempotent upserts.
    """
    from psycopg2 import sql as pgsql

    print(f"[train] Generating Top-{TOP_N} recommendations for all known users ...")

    # Build integer → raw string reverse-maps
    user_labels    = encoder.stages[0].labels
    product_labels = encoder.stages[1].labels
    user_map    = spark.createDataFrame(
        [(i, user_labels[i])    for i in range(len(user_labels))],
        ["userId", "raw_user"],
    )
    product_map = spark.createDataFrame(
        [(i, product_labels[i]) for i in range(len(product_labels))],
        ["productId", "raw_product"],
    )

    all_recs = best_model.recommendForAllUsers(TOP_N)
    result = (
        all_recs
        .select(col("userId"), posexplode(col("recommendations")).alias("rank", "rec"))
        .select(
            col("userId"),
            (col("rank") + 1).alias("rank"),
            col("rec.productId").alias("productId"),
            col("rec.rating").alias("predicted_rating"),
        )
        .join(user_map,    on="userId",    how="left")
        .join(product_map, on="productId", how="left")
        .select(
            col("raw_user").alias("user_id"),
            col("raw_product").alias("product_id"),
            col("rank"),
            col("predicted_rating"),
        )
    )

    # Write to a staging table via JDBC, then upsert atomically via psycopg2
    staging = "recs_from_training"
    result.write.jdbc(url=JDBC_URL, table=staging, mode="overwrite", properties=JDBC_PROP)

    conn = psycopg2.connect(
        host=_PG_HOST, port=int(_PG_PORT), dbname=_PG_DB,
        user=_PG_USER, password=_PG_PASS,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                staging_id = pgsql.Identifier(staging)
                cur.execute(
                    pgsql.SQL("""
                        INSERT INTO recommendations (user_id, product_id, rank, predicted_rating)
                        SELECT user_id, product_id, rank, predicted_rating FROM {}
                        ON CONFLICT (user_id, product_id) DO UPDATE SET
                            rank             = EXCLUDED.rank,
                            predicted_rating = EXCLUDED.predicted_rating,
                            created_at       = now()
                    """).format(staging_id)
                )
                cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(staging_id))
    finally:
        conn.close()

    print(f"[train] All-user recommendations written to PostgreSQL.")


def train(spark: SparkSession):

    started_at = datetime.now(timezone.utc).isoformat()

    # ── 1. Ingest from Kafka ──────────────────────────────────────────────────
    df = load_from_kafka(spark)

    # ── 2. Clean & filter ─────────────────────────────────────────────────────
    df, row_count = filter_active(df)

    # ── 3. Encode string IDs → integers ──────────────────────────────────────
    encoded, encoder = encode(df)

    n_users    = encoded.select("userId").distinct().count()
    n_products = encoded.select("productId").distinct().count()
    print(f"[train] Unique users: {n_users:,}  |  Unique products: {n_products:,}")

    # ── 4. 80 / 10 / 10 split — fixed seed for reproducibility ───────────────
    train_df, val_df, test_df = encoded.randomSplit([0.8, 0.1, 0.1], seed=42)
    train_df.cache()
    val_df.cache()

    # ── 5. ALS with hyperparameter grid ──────────────────────────────────────
    als = ALS(
        userCol           = "userId",
        itemCol           = "productId",
        ratingCol         = "rating",
        coldStartStrategy = "drop",   # avoids NaN poisoning RMSE
        nonnegative       = True,
        implicitPrefs     = False,    # explicit star-rating feedback
        seed              = 42,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank,     [10, 20])
        .addGrid(als.regParam, [0.01, 0.1])
        .addGrid(als.maxIter,  [10])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName    = "rmse",
        labelCol      = "rating",
        predictionCol = "prediction",
    )

    best_model  = None
    best_rmse   = float("inf")
    best_params = {}

    for params in param_grid:
        als.setParams(**{p.name: v for p, v in params.items()})
        m     = als.fit(train_df)
        preds = m.transform(val_df)
        rmse  = evaluator.evaluate(preds)
        print(f"[train] rank={params[als.rank]} reg={params[als.regParam]} → val RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = m
            best_params = {p.name: v for p, v in params.items()}

    print(f"[train] Best params: {best_params}  |  Val RMSE: {best_rmse:.4f}")

    # ── 6. Evaluate on held-out test set (10% never seen during training) ─────
    test_preds = best_model.transform(test_df)
    test_rmse  = evaluator.evaluate(test_preds)
    print(f"[train] Test RMSE (held-out 10%): {test_rmse:.4f}")

    # ── 7. Persist — wipe stale directories first ─────────────────────────────
    for stale in ["encoders", "als", "test_data"]:
        p = os.path.join(MODEL_PATH, stale)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    encoder.write().overwrite().save(f"{MODEL_PATH}/encoders")
    best_model.write().overwrite().save(f"{MODEL_PATH}/als")
    test_df.write.mode("overwrite").parquet(f"{MODEL_PATH}/test_data")

    metrics = {
        "val_rmse":        best_rmse,
        "test_rmse":       test_rmse,
        "best_params":     best_params,
        "trained_at":      started_at,
        "finished_at":     datetime.now(timezone.utc).isoformat(),
        "train_rows":      row_count,
        "unique_users":    n_users,
        "unique_products": n_products,
    }
    with open(f"{MODEL_PATH}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ── 8. Log model run to PostgreSQL ────────────────────────────────────────
    try:
        conn = psycopg2.connect(
            host=_PG_HOST, port=int(_PG_PORT), dbname=_PG_DB,
            user=_PG_USER, password=_PG_PASS
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_runs
                        (val_rmse, test_rmse, best_rank, best_reg, best_iter, train_rows)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        best_rmse,
                        test_rmse,
                        best_params.get("rank"),
                        best_params.get("regParam"),
                        best_params.get("maxIter"),
                        row_count,
                    ),
                )
        conn.close()
        print("[train] Model run logged to model_runs table.")
    except Exception as exc:
        print(f"[train] Warning: could not log to model_runs: {exc}")

    print(f"[train] Model saved to {MODEL_PATH}")

    # ── 9. Compute popular products (cold-start fallback for stream.py) ────────
    try:
        compute_popular_products(df)
    except Exception as exc:
        print(f"[train] Warning: could not compute popular products: {exc}")

    # ── 10. Write Top-N recs for ALL known users to PostgreSQL (Option B) ──────
    try:
        write_all_recommendations(spark, best_model, encoder)
    except Exception as exc:
        print(f"[train] Warning: could not write all-user recommendations: {exc}")

    return encoder, best_model


if __name__ == "__main__":
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    train(spark)
    spark.stop()
