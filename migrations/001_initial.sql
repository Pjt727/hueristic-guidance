-- Inference sessions — one row per user request
CREATE TABLE IF NOT EXISTS inference_sessions (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    obfuscated_id TEXT     NOT NULL,
    prompt        TEXT     NOT NULL,
    status        TEXT     NOT NULL DEFAULT 'pending', -- pending | streaming | complete | error
    result_text   TEXT,
    created_at    TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at  TEXT
);

-- Individual token events for each session
CREATE TABLE IF NOT EXISTS inference_tokens (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER  NOT NULL REFERENCES inference_sessions(id) ON DELETE CASCADE,
    position    INTEGER  NOT NULL,
    token_text  TEXT     NOT NULL,
    token_id    INTEGER  NOT NULL,
    probability REAL     NOT NULL,
    logit       REAL     NOT NULL,
    created_at  TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_tokens_session ON inference_tokens(session_id, position);
