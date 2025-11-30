-- Migration: Add semantic layout and clustering columns to chunks table
-- Date: 2025-11-15
-- Purpose: Support cinematic layout with UMAP coordinates, clustering, and enrichment metadata

-- Add missing columns to chunks table
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS umap_coordinates JSONB,
ADD COLUMN IF NOT EXISTS cluster_id INTEGER,
ADD COLUMN IF NOT EXISTS cluster_confidence REAL,
ADD COLUMN IF NOT EXISTS cluster_label TEXT,
ADD COLUMN IF NOT EXISTS nearest_chunk_ids JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS timestamp_created TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS timestamp_modified TIMESTAMPTZ DEFAULT NOW();

-- Convert legacy shape_3d column to JSONB for deterministic geometry payloads
ALTER TABLE chunks
ALTER COLUMN shape_3d DROP DEFAULT,
ALTER COLUMN shape_3d TYPE JSONB USING (
    CASE
        WHEN shape_3d IS NULL OR trim(shape_3d::text) = '' THEN jsonb_build_object('type', 'sphere')
        WHEN left(trim(shape_3d::text), 1) = '{' THEN shape_3d::jsonb
        ELSE jsonb_build_object('type', shape_3d::text)
    END
),
ALTER COLUMN shape_3d SET DEFAULT jsonb_build_object('type', 'sphere');

-- Create indexes for new columns to improve query performance
CREATE INDEX IF NOT EXISTS idx_chunks_cluster ON chunks(cluster_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(timestamp_created DESC);

-- Cleanup: Update stale enrichment entries
-- Set any chunks with pending_enrichment status to have NULL cluster data (will be re-enriched)
UPDATE chunks
SET
    cluster_id = NULL,
    cluster_confidence = NULL,
    cluster_label = NULL,
    umap_coordinates = NULL
WHERE metadata->>'status' = 'pending_enrichment';

-- Log migration completion
SELECT 'Migration completed: Added semantic layout columns to chunks table' AS status;
