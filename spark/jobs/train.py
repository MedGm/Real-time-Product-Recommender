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
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, count, desc, first, from_json, lit, mean, posexplode
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

# ── Kafka config ──────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "amazon-reviews")
CSV_PATH        = os.getenv("CSV_PATH",        "/dataset/Reviews.csv")

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
    StructField("UserId",                 StringType(), True),
    StructField("ProductId",              StringType(), True),
    StructField("Score",                  FloatType(),  True),
    StructField("Time",                   LongType(),   True),
    StructField("HelpfulnessNumerator",   LongType(),   True),
    StructField("HelpfulnessDenominator", LongType(),   True),
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
            col("d.HelpfulnessNumerator").alias("h_num"),
            col("d.HelpfulnessDenominator").alias("h_den"),
        )
        .dropna(subset=["raw_user", "raw_product", "rating"])
        .filter((col("rating") >= 1) & (col("rating") <= 5))
        # Keep reviews with no votes yet OR community helpfulness ratio >= 60 %.
        # Removes ~15 % of low-quality reviews that the community flagged as unhelpful.
        .filter(
            col("h_den").isNull() | (col("h_den") == 0) |
            (col("h_num") / col("h_den") >= 0.6)
        )
        .dropDuplicates(["raw_user", "raw_product"])
        .select("raw_user", "raw_product", "rating")
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


def write_all_recommendations(spark: SparkSession, best_model, encoder,
                              global_mean: float, user_biases, item_biases):
    """
    Generate Top-N recommendations for ALL known users using the trained ALS model
    and upsert them to PostgreSQL.

    ALS was trained on bias-corrected residuals, so predicted ratings are residuals.
    Final predicted_rating = residual + global_mean + user_bias + item_bias.
    """
    from psycopg2 import sql as pgsql

    print(f"[train] Generating Top-{TOP_N} recommendations for all known users ...")

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
            col("rec.rating").alias("residual"),
        )
        .join(user_map,    on="userId",    how="left")
        .join(product_map, on="productId", how="left")
        .join(user_biases, on="userId",    how="left")
        .join(item_biases, on="productId", how="left")
        .select(
            col("raw_user").alias("user_id"),
            col("raw_product").alias("product_id"),
            col("rank"),
            (col("residual") + lit(global_mean)
             + coalesce(col("user_bias"), lit(0.0))
             + coalesce(col("item_bias"), lit(0.0))).alias("predicted_rating"),
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


def save_product_names(spark: SparkSession):
    """
    Read Reviews.csv, extract a display name per product (most-helpful review
    Summary), and upsert into the postgres products table for the dashboard.
    """
    if not os.path.exists(CSV_PATH):
        print(f"[train] CSV not found at {CSV_PATH}, skipping product names.")
        return

    print("[train] Extracting product display names from CSV ...")
    csv_df = (
        spark.read
        .option("header", "true")
        .option("escape", '"')
        .csv(CSV_PATH)
        .select(
            col("ProductId").alias("product_id"),
            col("Summary").alias("summary"),
        )
        .dropna(subset=["product_id", "summary"])
        .filter(col("summary") != "")
    )

    products_df = (
        csv_df
        .groupBy("product_id")
        .agg(
            count("*").alias("review_count"),
            first("summary").alias("display_name"),
        )
    )

    rows = products_df.collect()
    conn = psycopg2.connect(
        host=_PG_HOST, port=int(_PG_PORT), dbname=_PG_DB,
        user=_PG_USER, password=_PG_PASS,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE products")
                cur.executemany(
                    """
                    INSERT INTO products (product_id, display_name, review_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (product_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        review_count = EXCLUDED.review_count
                    """,
                    [(r.product_id, r.display_name[:200], r.review_count) for r in rows],
                )
    finally:
        conn.close()
    print(f"[train] Saved {len(rows):,} product names to postgres.")


def train(spark: SparkSession):

    started_at = datetime.now(timezone.utc).isoformat()

    # ── 1. Ingest from Kafka ──────────────────────────────────────────────────
    df = load_from_kafka(spark)

    # ── 2. Clean & filter ─────────────────────────────────────────────────────
    df, _ = filter_active(df)

    # ── 3. Encode string IDs → integers ──────────────────────────────────────
    encoded, encoder = encode(df)

    n_users    = encoded.select("userId").distinct().count()
    n_products = encoded.select("productId").distinct().count()
    print(f"[train] Unique users: {n_users:,}  |  Unique products: {n_products:,}")

    # ── 4. 80 / 10 / 10 split on raw ratings — BEFORE bias computation ──────────
    # Split must come first: computing biases on the full dataset would leak
    # val/test rating information into the bias terms, artificially deflating
    # test residuals and making RMSE look better than it is.
    train_raw, val_raw, test_raw = encoded.randomSplit([0.8, 0.1, 0.1], seed=42)
    train_raw.cache()

    # ── 3.5. Bias decomposition — train split only ────────────────────────────
    # Decompose: r = μ + b_u + b_i + ε  (ALS trains on ε)
    # All statistics derived exclusively from training rows.
    global_mean = train_raw.select(mean("rating")).collect()[0][0]
    print(f"[train] Global mean rating (train only): {global_mean:.4f}")

    user_biases = (
        train_raw.groupBy("userId")
        .agg((mean("rating") - lit(global_mean)).alias("user_bias"))
    )
    item_biases = (
        train_raw.groupBy("productId")
        .agg((mean("rating") - lit(global_mean)).alias("item_bias"))
    )
    user_biases.cache()
    item_biases.cache()

    def apply_biases(df):
        """Apply train-derived biases; unknown users/items in val/test get 0."""
        return (
            df
            .join(user_biases, "userId",    "left")
            .join(item_biases, "productId", "left")
            .withColumn(
                "residual",
                col("rating") - lit(global_mean)
                - coalesce(col("user_bias"), lit(0.0))
                - coalesce(col("item_bias"), lit(0.0)),
            )
        )

    train_df = apply_biases(train_raw)
    val_df   = apply_biases(val_raw)
    test_df  = apply_biases(test_raw)
    train_raw.unpersist()

    train_df.cache()
    val_df.cache()
    train_count = train_df.count()
    print(f"[train] Split → train: {train_count:,}  val: {val_df.count():,}  test: {test_df.count():,}")

    # ── 5. ALS hyperparameter search via TrainValidationSplit (Spark MLlib) ──
    als = ALS(
        userCol           = "userId",
        itemCol           = "productId",
        ratingCol         = "residual",   # train on bias-corrected residuals
        coldStartStrategy = "drop",
        nonnegative       = False,        # residuals can be negative
        implicitPrefs     = False,
        seed              = 42,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank,     [20, 50, 100])
        .addGrid(als.regParam, [0.05, 0.1])
        .addGrid(als.maxIter,  [15])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName    = "rmse",
        labelCol      = "residual",
        predictionCol = "prediction",
    )

    # TrainValidationSplit fits every param combination and selects the best
    # model using an internal 80/20 split of train_df — the standard Spark MLlib
    # approach for hyperparameter tuning without k-fold overhead.
    tvs = TrainValidationSplit(
        estimator          = als,
        estimatorParamMaps = param_grid,
        evaluator          = evaluator,
        trainRatio         = 0.8,
        seed               = 42,
        parallelism        = 2,
    )

    print("[train] Running hyperparameter search via TrainValidationSplit ...")
    tvs_model  = tvs.fit(train_df)
    best_model = tvs_model.bestModel

    # Log all param combinations and their validation RMSE
    for params, metric in zip(param_grid, tvs_model.validationMetrics):
        print(f"[train]   rank={params[als.rank]} reg={params[als.regParam]} iter={params[als.maxIter]} → RMSE={metric:.4f}")

    best_idx    = tvs_model.validationMetrics.index(min(tvs_model.validationMetrics))
    best_params = {p.name: v for p, v in param_grid[best_idx].items()}

    # Evaluate best model on the independent val_df (not seen during TVS search)
    val_preds = best_model.transform(val_df)
    best_rmse = evaluator.evaluate(val_preds)
    print(f"[train] Best params: {best_params}  |  Val RMSE: {best_rmse:.4f}")

    # Release cached DataFrames — no longer needed after tuning
    train_df.unpersist()
    val_df.unpersist()
    user_biases.unpersist()
    item_biases.unpersist()

    # ── 6. Evaluate on held-out test set (10% never seen during training) ─────
    test_preds = best_model.transform(test_df)
    test_rmse  = evaluator.evaluate(test_preds)
    print(f"[train] Test RMSE (held-out 10%): {test_rmse:.4f}")

    # ── 7. Persist — wipe stale directories first ─────────────────────────────
    for stale in ["encoders", "als", "test_data", "user_biases", "item_biases"]:
        p = os.path.join(MODEL_PATH, stale)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    encoder.write().overwrite().save(f"{MODEL_PATH}/encoders")
    best_model.write().overwrite().save(f"{MODEL_PATH}/als")
    test_df.write.mode("overwrite").parquet(f"{MODEL_PATH}/test_data")
    user_biases.write.mode("overwrite").parquet(f"{MODEL_PATH}/user_biases")
    item_biases.write.mode("overwrite").parquet(f"{MODEL_PATH}/item_biases")

    metrics = {
        "val_rmse":        best_rmse,
        "test_rmse":       test_rmse,
        "best_params":     best_params,
        "trained_at":      started_at,
        "finished_at":     datetime.now(timezone.utc).isoformat(),
        "train_rows":      train_count,
        "unique_users":    n_users,
        "unique_products": n_products,
        "global_mean":     global_mean,
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
                        train_count,
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
        write_all_recommendations(spark, best_model, encoder,
                                  global_mean, user_biases, item_biases)
    except Exception as exc:
        print(f"[train] Warning: could not write all-user recommendations: {exc}")

    # ── 11. Save product display names (for dashboard) ────────────────────────
    try:
        save_product_names(spark)
    except Exception as exc:
        print(f"[train] Warning: could not save product names: {exc}")

    return encoder, best_model


if __name__ == "__main__":
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    train(spark)
    spark.stop()
