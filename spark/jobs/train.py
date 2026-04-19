"""
Spark batch job — trains an ALS model on Amazon Fine Food Reviews.

Splits:
  80% train | 10% validation (hyperparameter tuning) | 10% test (held-out)

Outputs:
  /model/als          — saved ALS PipelineModel
  /model/encoders     — saved StringIndexer models (user + product)
  /model/test_data    — parquet of the 10% held-out set (for offline eval)

Writes RMSE values to stdout and to /model/metrics.json.
"""

import json
import os
import shutil
from datetime import datetime, timezone

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

CSV_PATH   = os.getenv("CSV_PATH", "/dataset/Reviews.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/model")

# Filtering thresholds — drop rare users / items
MIN_USER_RATINGS    = int(os.getenv("MIN_USER_RATINGS", "5"))
MIN_PRODUCT_RATINGS = int(os.getenv("MIN_PRODUCT_RATINGS", "5"))


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ALS-Training")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def load_and_clean(spark: SparkSession):
    df = (
        spark.read.csv(CSV_PATH, header=True, inferSchema=True)
        .select(
            col("UserId").alias("raw_user"),
            col("ProductId").alias("raw_product"),
            col("Score").cast("float").alias("rating"),
        )
        .dropna()
        .filter((col("rating") >= 1) & (col("rating") <= 5))
    )

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


def train(spark: SparkSession):
    started_at = datetime.now(timezone.utc).isoformat()
    df, row_count = load_and_clean(spark)
    encoded, encoder = encode(df)

    # Count unique users / products after encoding
    n_users    = encoded.select("userId").distinct().count()
    n_products = encoded.select("productId").distinct().count()
    print(f"[train] Unique users: {n_users:,}  |  Unique products: {n_products:,}")

    # 80 / 10 / 10 split — fixed seed for reproducibility
    train_df, val_df, test_df = encoded.randomSplit([0.8, 0.1, 0.1], seed=42)
    train_df.cache()
    val_df.cache()

    als = ALS(
        userCol           = "userId",
        itemCol           = "productId",
        ratingCol         = "rating",
        coldStartStrategy = "drop",
        nonnegative       = True,
        implicitPrefs     = False,  # explicit star-rating feedback
        seed              = 42,
    )

    # Hyperparameter grid (manually evaluated on val_df)
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

    # Evaluate on held-out test set (10% never seen during training)
    test_preds = best_model.transform(test_df)
    test_rmse  = evaluator.evaluate(test_preds)
    print(f"[train] Test RMSE (held-out 10%): {test_rmse:.4f}")

    # Persist — wipe stale directories first to avoid Spark ML overwrite issues
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

    print(f"[train] Model saved to {MODEL_PATH}")
    return encoder, best_model


if __name__ == "__main__":
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    train(spark)
    spark.stop()
