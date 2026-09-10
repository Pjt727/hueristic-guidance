-- Classifier comparison runs and results.
-- Used by POST /classify/compare to store multi-method evaluation results.

CREATE TABLE IF NOT EXISTS classifier_comparison_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    INTEGER NOT NULL,
    methods     TEXT    NOT NULL,  -- JSON array of method names used
    started_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT,
    total       INTEGER,
    summary     TEXT              -- JSON: per-method accuracy summary
);

CREATE TABLE IF NOT EXISTS classifier_comparison_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES classifier_comparison_runs(id) ON DELETE CASCADE,
    example_id  INTEGER NOT NULL,
    example_text TEXT   NOT NULL,
    correct_categories TEXT NOT NULL,  -- JSON array of correct category names
    results     TEXT    NOT NULL       -- JSON object: { "method_name": { "chosen": "...", "scores": [...], "confidence": N, "latency_ms": N } }
);

CREATE INDEX IF NOT EXISTS idx_comparison_results_run ON classifier_comparison_results(run_id);
