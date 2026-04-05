"""
Spark Structured Streaming job — consumes amazon-reviews from Kafka,
generates Top-N recommendations for each new user seen, and writes
results to PostgreSQL.

Requires the trained model produced by train.py to be present at MODEL_PATH.
"""

import os

from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "amazon-reviews")
MODEL_PATH      = os.getenv("MODEL_PATH", "/model")
DB_URL          = os.getenv("DB_URL", "postgresql://bigdata:bigdata@postgres/recommendations")
TOP_N           = int(os.getenv("TOP_N", "10"))

# JDBC url for Spark (different format from SQLAlchemy)
JDBC_URL  = "jdbc:postgresql://postgres:5432/recommendations"
JDBC_PROP = {"user": "bigdata", "password": "bigdata", "driver": "org.postgresql.Driver"}

EVENT_SCHEMA = StructType([
    StructField("UserId",    StringType(), True),
    StructField("ProductId", StringType(), True),
    StructField("Score",     FloatType(),  True),
    StructField("Time",      LongType(),   True),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ALS-Streaming")
        .config("spark.sql.shuffle.partitions", "10")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.postgresql:postgresql:42.7.3",
        )
        .getOrCreate()
    )


def load_models():
    encoder   = PipelineModel.load(f"{MODEL_PATH}/encoders")
    als_model = ALSModel.load(f"{MODEL_PATH}/als")
    return encoder, als_model


def write_recommendations(batch_df, batch_id):
    """Micro-batch handler: encode new users, generate Top-N, upsert to Postgres."""
    if batch_df.rdd.isEmpty():
        return

    spark = batch_df.sparkSession
    encoder, als_model = load_models()  # reload each batch (cheap — already cached by OS)

    # Encode only user IDs using the user StringIndexer (stage 0).
    # Do NOT run the full pipeline — the product indexer would drop rows for
    # unknown product IDs (handleInvalid="skip"), giving zero results.
    user_indexer = encoder.stages[0]
    encoded = user_indexer.transform(
        batch_df.select(col("UserId").alias("raw_user")).dropDuplicates(["raw_user"])
    ).select(col("userId").cast("int"), col("raw_user"))

    # ALS recommendForUserSubset expects a DataFrame with userId column
    user_subset = encoded.select(col("userId"))
    recs = als_model.recommendForUserSubset(user_subset, TOP_N)

    # recs schema: userId, recommendations:[{productId, rating}]
    from pyspark.sql.functions import explode, posexplode
    from pyspark.sql.types import ArrayType, IntegerType

    exploded = recs.select(
        col("userId"),
        posexplode(col("recommendations")).alias("rank", "rec"),
    ).select(
        col("userId"),
        (col("rank") + 1).alias("rank"),
        col("rec.productId").alias("productId"),
        col("rec.rating").alias("predicted_rating"),
    )

    # Join back to get raw user/product strings
    user_map = encoded.select(col("userId"), col("raw_user"))

    # We need a product reverse-map; load it from the saved StringIndexerModel labels
    # The second stage of the pipeline encodes products
    product_labels = encoder.stages[1].labels  # array of original product IDs
    product_map_rows = [(i, product_labels[i]) for i in range(len(product_labels))]
    product_map = spark.createDataFrame(product_map_rows, ["productId", "raw_product"])

    result = (
        exploded
        .join(user_map,    on="userId",    how="left")
        .join(product_map, on="productId", how="left")
        .select(
            col("raw_user").alias("user_id"),
            col("raw_product").alias("product_id"),
            col("rank"),
            col("predicted_rating"),
        )
    )

    # Write to Postgres (upsert via overwrite on a temp staging table pattern)
    result.write.jdbc(
        url=JDBC_URL,
        table="recommendations",
        mode="append",
        properties=JDBC_PROP,
    )
    print(f"[stream] Batch {batch_id}: wrote {result.count()} recommendation rows.")


def run():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    events = (
        raw_stream
        .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("data"))
        .select("data.*")
        .filter(col("UserId").isNotNull())
    )

    query = (
        events.writeStream
        .foreachBatch(write_recommendations)
        .option("checkpointLocation", f"{MODEL_PATH}/stream_checkpoint")
        .trigger(processingTime="30 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
