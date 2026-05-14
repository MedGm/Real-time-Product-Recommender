"""One-shot script: read user_biases and item_biases parquet → write to postgres."""
import os
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

MODEL_PATH = os.getenv("MODEL_PATH", "/model")
PG_HOST    = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT    = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB      = os.getenv("POSTGRES_DB",   "recommendations")
PG_USER    = os.getenv("POSTGRES_USER", "bigdata")
PG_PASS    = os.getenv("POSTGRES_PASSWORD", "bigdata")

JDBC_URL  = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROP = {"user": PG_USER, "password": PG_PASS, "driver": "org.postgresql.Driver"}

spark = (
    SparkSession.builder
    .appName("load-biases")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

encoder = None
try:
    from pyspark.ml import PipelineModel
    encoder = PipelineModel.load(f"{MODEL_PATH}/encoders")
    user_labels    = encoder.stages[0].labels
    product_labels = encoder.stages[1].labels
except Exception as e:
    print(f"[load_biases] Could not load encoder: {e}")
    spark.stop()
    raise SystemExit(1)

user_map = spark.createDataFrame(
    [(i, user_labels[i]) for i in range(len(user_labels))],
    ["userId", "user_id"]
)
product_map = spark.createDataFrame(
    [(i, product_labels[i]) for i in range(len(product_labels))],
    ["productId", "product_id"]
)

ub = spark.read.parquet(f"{MODEL_PATH}/user_biases")
ib = spark.read.parquet(f"{MODEL_PATH}/item_biases")

ub_str = ub.join(user_map, on="userId", how="left").select("user_id", "user_bias")
ib_str = ib.join(product_map, on="productId", how="left").select("product_id", "item_bias")

ub_str.write.jdbc(url=JDBC_URL, table="user_biases_staging", mode="overwrite", properties=JDBC_PROP)
ib_str.write.jdbc(url=JDBC_URL, table="item_biases_staging", mode="overwrite", properties=JDBC_PROP)

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
with conn:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_biases (user_id, user_bias)
            SELECT user_id, user_bias FROM user_biases_staging
            ON CONFLICT (user_id) DO UPDATE SET user_bias = EXCLUDED.user_bias
        """)
        cur.execute("DROP TABLE IF EXISTS user_biases_staging")
        cur.execute("""
            INSERT INTO item_biases (product_id, item_bias)
            SELECT product_id, item_bias FROM item_biases_staging
            ON CONFLICT (product_id) DO UPDATE SET item_bias = EXCLUDED.item_bias
        """)
        cur.execute("DROP TABLE IF EXISTS item_biases_staging")
conn.close()

ub_count = conn = None
conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
with conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM user_biases")
        ub_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM item_biases")
        ib_count = cur.fetchone()[0]
conn.close()

print(f"[load_biases] Done. user_biases: {ub_count:,}  item_biases: {ib_count:,}")
spark.stop()
