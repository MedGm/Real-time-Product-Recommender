"""One-shot: recompute product display names using most-helpful review summary."""
import os
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, length, row_number, when
from pyspark.sql.window import Window

CSV_PATH = os.getenv("CSV_PATH", "/dataset/Reviews.csv")
PG_HOST  = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT  = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB    = os.getenv("POSTGRES_DB",   "recommendations")
PG_USER  = os.getenv("POSTGRES_USER", "bigdata")
PG_PASS  = os.getenv("POSTGRES_PASSWORD", "bigdata")

spark = SparkSession.builder.appName("fix-product-names").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

csv_df = (
    spark.read
    .option("header", "true")
    .option("escape", '"')
    .csv(CSV_PATH)
    .select(
        col("ProductId").alias("product_id"),
        col("Summary").alias("summary"),
        col("HelpfulnessNumerator").cast("int").alias("h_num"),
        col("HelpfulnessDenominator").cast("int").alias("h_den"),
    )
    .dropna(subset=["product_id", "summary"])
    .filter(col("summary") != "")
    .withColumn("h_ratio", when(col("h_den") > 0, col("h_num") / col("h_den")).otherwise(0.0))
)

w = Window.partitionBy("product_id").orderBy(
    desc("h_ratio"), desc("h_num"), desc(length("summary"))
)
best = (
    csv_df
    .withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .select("product_id", "summary")
)

products_df = (
    csv_df.groupBy("product_id").agg(count("*").alias("review_count"))
    .join(best, on="product_id", how="left")
    .select(col("product_id"), col("summary").alias("display_name"), col("review_count"))
)

rows = products_df.collect()
print(f"[fix_product_names] Collected {len(rows):,} products")

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
with conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE products")
        cur.executemany(
            "INSERT INTO products (product_id, display_name, review_count) VALUES (%s, %s, %s)",
            [(r.product_id, (r.display_name or "")[:200], r.review_count) for r in rows],
        )
conn.close()

print(f"[fix_product_names] Done. {len(rows):,} products saved.")
spark.stop()
