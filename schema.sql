CREATE EXTENSION IF NOT EXISTS vector;

DO $$ BEGIN
    CREATE TYPE event_type_enum AS ENUM ('entry', 'exit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE write_status_enum AS ENUM ('pending', 'complete');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    video_source TEXT NOT NULL,
    unique_visitors INTEGER NOT NULL DEFAULT 0,
    config_snapshot JSONB
);

CREATE TABLE IF NOT EXISTS faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding vector(512) NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    visit_count INTEGER NOT NULL DEFAULT 1,
    thumbnail_path TEXT,
    embedding_version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    face_id UUID REFERENCES faces(id),
    event_type event_type_enum NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    image_path TEXT NOT NULL,
    write_status write_status_enum NOT NULL DEFAULT 'pending',
    track_id INTEGER,
    similarity_score FLOAT,
    frame_number INTEGER NOT NULL,
    session_id UUID REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_pending ON events(write_status) WHERE write_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_faces_last_seen ON faces(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_faces_embedding ON faces USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
