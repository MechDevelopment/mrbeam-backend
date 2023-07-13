CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE predictions(
    id UUID PRIMARY KEY NOT NULL DEFAULT (uuid_generate_v4()),
    created_at TIMESTAMP
    WITH
        TIME ZONE DEFAULT NOW(),
    prediction JSONB,
    correction JSONB,
    image TEXT
);
