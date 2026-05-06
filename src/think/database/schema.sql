CREATE TABLE IF NOT EXISTS think_schema (
    id                    SERIAL PRIMARY KEY,
    event_id              INTEGER,
    timestamp             FLOAT NOT NULL,

    -- sensor inputs
    triggered_sensors     JSONB,
    sensor_readings       JSONB,
    sensor_normalized     JSONB,

    -- vision inputs
    composite_label       TEXT,
    glimpsed_fire         BOOLEAN,
    human_near_fire       BOOLEAN,
    fire_count            INT,
    smoke_count           INT,
    fire_union_area       FLOAT,
    smoke_union_area      FLOAT,
    cluster_count         INT,
    scene_label           TEXT,
    scene_confidence      FLOAT,
    fire_clusters         JSONB,
    raw_detections        JSONB,
    frame_image_url       TEXT,

    -- think output
    danger_level          INT,
    danger_label          TEXT,
    recommended_action    TEXT,

    -- training
    validated             BOOLEAN DEFAULT FALSE,
    true_danger_level     INT,
    true_action           TEXT
);

CREATE INDEX IF NOT EXISTS idx_think_schema_event_id
    ON think_schema(event_id);