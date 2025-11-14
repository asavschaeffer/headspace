-- Supabase SQL Schema for Headspace
-- Run this in your Supabase SQL editor to create the tables
--
-- SETUP INSTRUCTIONS:
-- 1. Go to your Supabase project dashboard
-- 2. Navigate to SQL Editor
-- 3. Run this entire script
-- 4. Optional: Enable pgvector extension for better vector operations:
--    CREATE EXTENSION IF NOT EXISTS vector;
--    Then uncomment the vector column in chunks table below

-- Optional: Enable pgvector extension for better embedding operations
-- Uncomment the next line if you want to use pgvector (recommended for production)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    doc_type TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
    -- Note: No foreign key to auth.users since user_id is TEXT (supports "anonymous" and custom IDs)
    -- Row Level Security (RLS) policies below handle user isolation
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_type TEXT,
    -- Embedding storage: JSONB (works out of the box) or vector (if pgvector enabled)
    -- JSONB format: stores as array of floats, e.g., [0.1, 0.2, ...]
    embedding JSONB,
    -- Uncomment the line below if you enabled pgvector extension above
    -- embedding_vector vector(384),  -- Adjust dimension based on your embedding model
    position_3d JSONB,
    color TEXT,
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    tag_confidence JSONB DEFAULT '{}',
    reasoning TEXT,
    shape_3d TEXT DEFAULT 'sphere',
    texture TEXT DEFAULT 'smooth',
    embedding_model TEXT DEFAULT '',  -- Track which model generated the embedding
    CONSTRAINT fk_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    -- Note: No foreign key to auth.users since user_id is TEXT (supports "anonymous" and custom IDs)
    -- Row Level Security (RLS) policies below handle user isolation
);

-- Connections table
CREATE TABLE IF NOT EXISTS connections (
    from_chunk_id TEXT NOT NULL,
    to_chunk_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    connection_type TEXT,
    strength REAL DEFAULT 1.0,
    PRIMARY KEY (from_chunk_id, to_chunk_id),
    CONSTRAINT fk_from_chunk FOREIGN KEY (from_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    CONSTRAINT fk_to_chunk FOREIGN KEY (to_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    -- Note: No foreign key to auth.users since user_id is TEXT (supports "anonymous" and custom IDs)
    -- Row Level Security (RLS) policies below handle user isolation
);

-- Attachments table
CREATE TABLE IF NOT EXISTS attachments (
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    attachment_type TEXT DEFAULT 'document',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (chunk_id, document_id),
    CONSTRAINT fk_chunk FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    CONSTRAINT fk_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    -- Note: No foreign key to auth.users since user_id is TEXT (supports "anonymous" and custom IDs)
    -- Row Level Security (RLS) policies below handle user isolation
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_connections_from ON connections(from_chunk_id);
CREATE INDEX IF NOT EXISTS idx_connections_to ON connections(to_chunk_id);
CREATE INDEX IF NOT EXISTS idx_attachments_chunk ON attachments(chunk_id);

-- Optional: Create vector index for similarity search (if using pgvector)
-- Uncomment if you enabled pgvector extension above
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_vector ON chunks 
-- USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);

-- Enable Row Level Security (RLS)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;

-- Create policies for RLS
-- Users can only access their own data
-- Note: These policies work with Supabase Auth (auth.uid()) or custom text user_ids
-- For anonymous/custom user_ids, you may need to adjust these policies or disable RLS
CREATE POLICY "Users can view own documents" ON documents
    FOR SELECT USING (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can insert own documents" ON documents
    FOR INSERT WITH CHECK (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can update own documents" ON documents
    FOR UPDATE USING (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can delete own documents" ON documents
    FOR DELETE USING (auth.uid()::text = user_id OR user_id = 'anonymous');

-- Similar policies for chunks, connections, attachments
CREATE POLICY "Users can manage own chunks" ON chunks
    FOR ALL USING (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can manage own connections" ON connections
    FOR ALL USING (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can manage own attachments" ON attachments
    FOR ALL USING (auth.uid()::text = user_id OR user_id = 'anonymous');

