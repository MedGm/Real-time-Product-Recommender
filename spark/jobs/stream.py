"""
Spark Structured Streaming job — consumes amazon-reviews from Kafka,
generates Top-N recommendations for each new user seen, and upserts
results to PostgreSQL.

Models are loaded ONCE at startup (not per micro-batch) for performance.
Requires the trained model produced by train.py to be present at MODEL_PATH.
"""

import os

import psycopg2
from psycopg2 import sql as pgsql
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, posexplode
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "amazon-reviews")
MODEL_PATH      = os.getenv("MODEL_PATH", "/model")
TOP_N           = int(os.getenv("TOP_N", "10"))

_PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB",   "recommendations")
_PG_USER = os.getenv("POSTGRES_USER", "bigdata")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "bigdata")

JDBC_URL  = f"jdbc:postgresql://{_PG_HOST}:{_PG_PORT}/{_PG_DB}"
JDBC_PROP = {"user": _PG_USER, "password": _PG_PASS, "driver": "org.postgresql.Driver"}

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
        .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "false")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.postgresql:postgresql:42.7.3",
        )
        .getOrCreate()
    )


def load_models():
    """Load encoder pipeline and ALS model from the shared volume."""
    encoder   = PipelineModel.load(f"{MODEL_PATH}/encoders")
    als_model = ALSModel.load(f"{MODEL_PATH}/als")
    return encoder, als_model


def make_batch_handler(spark: SparkSession, encoder: PipelineModel, als_model: ALSModel):
    """
    Return a foreachBatch function that closes over the pre-loaded models.
    Models are loaded once at startup instead of on every micro-batch.
    """
    # Build the product reverse-map once (label index → raw product ID string)
    product_labels   = encoder.stages[1].labels
    product_map_rows = [(i, product_labels[i]) for i in range(len(product_labels))]
    product_map      = spark.createDataFrame(product_map_rows, ["productId", "raw_product"])
    product_map.cache()

    user_indexer = encoder.stages[0]

    def write_recommendations(batch_df, batch_id):
        """Micro-batch handler: encode new users, generate Top-N, upsert to Postgres."""
        if batch_df.rdd.isEmpty():
            return

        # Encode unique user IDs seen in this batch
        encoded = user_indexer.transform(
            batch_df.select(col("UserId").alias("raw_user")).dropDuplicates(["raw_user"])
        ).select(col("userId").cast("int"), col("raw_user"))

        # ALS recommendForUserSubset
        user_subset = encoded.select(col("userId"))
        recs        = als_model.recommendForUserSubset(user_subset, TOP_N)

        exploded = recs.select(
            col("userId"),
            posexplode(col("recommendations")).alias("rank", "rec"),
        ).select(
            col("userId"),
            (col("rank") + 1).alias("rank"),
            col("rec.productId").alias("productId"),
            col("rec.rating").alias("predicted_rating"),
        )

        user_map = encoded.select(col("userId"), col("raw_user"))

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

        # Upsert via staging table: write to temp, then INSERT ... ON CONFLICT DO UPDATE
        staging = f"recs_staging_{batch_id}"
        result.write.jdbc(url=JDBC_URL, table=staging, mode="overwrite", properties=JDBC_PROP)

        conn = psycopg2.connect(
            host=_PG_HOST, port=int(_PG_PORT), dbname=_PG_DB,
            user=_PG_USER, password=_PG_PASS
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

        print(f"[stream] Batch {batch_id}: upsert complete.")

    return write_recommendations


def run():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("[stream] Loading ALS model and encoders...")
    encoder, als_model = load_models()
    print("[stream] Models loaded. Starting streaming query...")

    batch_handler = make_batch_handler(spark, encoder, als_model)

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
        .foreachBatch(batch_handler)
        .option("checkpointLocation", f"{MODEL_PATH}/stream_checkpoint")
        .trigger(processingTime="30 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
