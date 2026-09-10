-- Column classifier_results was applied manually to app.db before this migration
-- was registered with sqlx. This is a no-op so sqlx can record it as applied.
-- The actual schema change is:
--   ALTER TABLE bulk_test_results ADD COLUMN classifier_results TEXT NOT NULL DEFAULT '[]';
SELECT 1 WHERE 0;
