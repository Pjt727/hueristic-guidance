-- Per-message kappa scaling constants for embedding-based logit adjustment.
-- message_id references vcmessages.id in the external Postgres VC database.
-- kappa defaults to 10.0 and can be tuned per message over time.
CREATE TABLE IF NOT EXISTS vc_message_constants (
    message_id INTEGER PRIMARY KEY,
    kappa      REAL    NOT NULL DEFAULT 10.0
);
