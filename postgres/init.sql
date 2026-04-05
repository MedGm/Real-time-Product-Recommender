-- Recommendations table
-- Populated by the Spark streaming job (stream.py)

CREATE TABLE IF NOT EXISTS recommendations (
    id               SERIAL PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    product_id       TEXT    NOT NULL,
    rank             INT     NOT NULL,   -- 1 = top recommendation
    predicted_rating FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups by user
CREATE INDEX IF NOT EXISTS idx_recs_user_id ON recommendations (user_id);

-- Unique constraint so we can upsert without duplicates
-- (user_id, product_id) pair should be unique per run;
-- for simplicity we allow multiple runs to append.
