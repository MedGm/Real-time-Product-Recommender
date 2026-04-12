-- ─────────────────────────────────────────────────────────────
-- Recommendations table
-- Populated by the Spark streaming job (stream.py)
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS recommendations (
    id               SERIAL PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    product_id       TEXT    NOT NULL,
    rank             INT     NOT NULL,        -- 1 = top recommendation
    predicted_rating FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),

    -- Unique per user/product pair — allows ON CONFLICT upsert
    CONSTRAINT uq_user_product UNIQUE (user_id, product_id)
);

-- Index for fast lookups by user
CREATE INDEX IF NOT EXISTS idx_recs_user_id ON recommendations (user_id);

-- ─────────────────────────────────────────────────────────────
-- Model training runs — stores metrics history
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_runs (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ DEFAULT NOW(),
    val_rmse     FLOAT,
    test_rmse    FLOAT,
    best_rank    INT,
    best_reg     FLOAT,
    best_iter    INT,
    train_rows   BIGINT,
    notes        TEXT
);
