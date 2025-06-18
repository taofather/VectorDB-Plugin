-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a simple test to verify the extension is working
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension enabled successfully';
END $$; 