# Supabase Setup Guide for Headspace

This guide will help you set up Supabase for your Headspace app to enable cloud storage and embedding data access.

## Prerequisites

- A Supabase account (free tier available at https://supabase.com)
- Your Headspace app codebase

## Step 1: Create a Supabase Project

1. Go to https://supabase.com and sign in (or create an account)
2. Click "New Project"
3. Fill in:
   - **Name**: Your project name (e.g., "headspace-app")
   - **Database Password**: Choose a strong password (save this!)
   - **Region**: Choose the closest region to your users
   - **Pricing Plan**: Free tier is fine for development
4. Click "Create new project"
5. Wait for the project to be created (takes 1-2 minutes)

## Step 2: Set Up the Database Schema

1. In your Supabase project dashboard, click on **SQL Editor** in the left sidebar
2. Click **New Query**
3. Open the `supabase_schema.sql` file from your project
4. Copy the entire contents and paste it into the SQL Editor
5. Click **Run** (or press Ctrl+Enter)
6. You should see "Success. No rows returned" - this means the tables were created successfully

### Optional: Enable pgvector Extension (Recommended for Production)

If you want better performance for embedding similarity searches:

1. In the SQL Editor, run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Then modify the `chunks` table to add a vector column:
   ```sql
   ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_vector vector(384);
   ```
   (Adjust the dimension `384` based on your embedding model's output size)
3. Create an index for faster similarity searches:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_chunks_embedding_vector ON chunks 
   USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
   ```

## Step 3: Get Your API Credentials

1. In your Supabase project dashboard, click on **Settings** (gear icon) in the left sidebar
2. Click on **API** in the settings menu
3. You'll see two important values:
   - **Project URL**: Copy this (looks like `https://xxxxx.supabase.co`)
   - **anon public key**: Copy this (long string starting with `eyJ...`)

## Step 4: Configure Your Environment Variables

1. Copy `env.example` to `.env` if you haven't already:
   ```bash
   cp env.example .env
   ```

2. Open `.env` and add your Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-anon-key-here
   ```

3. Set the storage mode (optional):
   ```env
   STORAGE_MODE=auto  # or "cloud" to force cloud mode
   ```

## Step 5: Test the Connection

You can test your Supabase connection by running:

```python
from headspace.services.supabase_storage import SupabaseStorage
import os

storage = SupabaseStorage(
    supabase_url=os.environ.get("SUPABASE_URL"),
    supabase_key=os.environ.get("SUPABASE_KEY")
)

if storage.test_connection():
    print("✅ Supabase connection successful!")
else:
    print("❌ Supabase connection failed. Check your credentials.")
```

## Step 6: Verify the Setup

1. In your Supabase dashboard, go to **Table Editor**
2. You should see these tables:
   - `documents`
   - `chunks`
   - `connections`
   - `attachments`

3. Try uploading a document in your Headspace app - it should now save to Supabase!

## Troubleshooting

### Issue: "relation does not exist" error
**Solution**: Make sure you ran the `supabase_schema.sql` script in the SQL Editor.

### Issue: "permission denied" error
**Solution**: Check that Row Level Security (RLS) policies are set up correctly. The schema includes RLS policies, but if you're using anonymous access, you may need to adjust them.

### Issue: Embeddings not saving/loading correctly
**Solution**: 
- Check that the `embedding` column in the `chunks` table is JSONB type
- Verify your embedding data is a list of floats (not numpy arrays)
- Check the console logs for serialization errors

### Issue: Connection timeout
**Solution**: 
- Verify your `SUPABASE_URL` is correct
- Check that your `SUPABASE_KEY` is the anon/public key (not the service role key)
- Ensure your network allows connections to Supabase

## Security Notes

- The `anon` key is safe to use in client-side code (it's public)
- Row Level Security (RLS) ensures users can only access their own data
- Never commit your `.env` file to version control
- For production, consider using environment variables on your hosting platform

## Next Steps

Once Supabase is set up:
- Your app will automatically use cloud storage when `SUPABASE_URL` and `SUPABASE_KEY` are set
- Embeddings will be stored and retrieved from Supabase
- Data will sync across devices (if using the same user_id)
- You can view and manage your data in the Supabase dashboard

## Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase/supabase-py)
- [pgvector Documentation](https://github.com/pgvector/pgvector) (for vector similarity search)

