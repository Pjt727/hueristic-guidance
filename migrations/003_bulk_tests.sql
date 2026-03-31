-- One row per bulk test run (agent + timestamp + summary stats)
CREATE TABLE IF NOT EXISTS bulk_test_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      INTEGER NOT NULL,
    started_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at  TEXT,
    total         INTEGER,
    success_count INTEGER
);

-- One row per example result within a bulk test run.
-- chosen_category  — NULL if inference errored
-- correct_categories — JSON array of strings
-- steps            — JSON array of StepCandidates (full token-level detail)
CREATE TABLE IF NOT EXISTS bulk_test_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES bulk_test_runs(id) ON DELETE CASCADE,
    example_id         INTEGER NOT NULL,
    example_text       TEXT    NOT NULL,
    chosen_category    TEXT,
    correct_categories TEXT    NOT NULL,
    success            INTEGER NOT NULL,
    steps              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bulk_results_run ON bulk_test_results(run_id);
