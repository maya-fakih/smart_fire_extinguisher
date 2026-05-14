-- src/notify/schema.sql
-- Notifications table — written by NotificationService, read by the website
-- and any audit/analysis tooling.

CREATE TABLE IF NOT EXISTS notifications (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_type      TEXT NOT NULL,         -- matches EventType enum values
    severity        TEXT NOT NULL,         -- info | warn | critical
    source_layer    TEXT,                  -- sense | see | think | act | orchestrator
    payload         JSONB,                 -- event-specific context
    acknowledged    BOOLEAN DEFAULT FALSE  -- user marked read on website
);

-- Index by severity so the website can list critical notifications fast.
CREATE INDEX IF NOT EXISTS idx_notifications_severity
    ON notifications(severity);

-- Index by timestamp DESC for the "recent activity" feed.
CREATE INDEX IF NOT EXISTS idx_notifications_timestamp_desc
    ON notifications(timestamp DESC);

-- Index by event_type so the website can filter by category.
CREATE INDEX IF NOT EXISTS idx_notifications_event_type
    ON notifications(event_type);