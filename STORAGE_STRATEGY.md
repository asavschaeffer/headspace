# Storage Strategy for Headspace

## Current Implementation

The app uses **auto-detection** to choose between local SQLite and Supabase cloud storage:

- **Local SQLite** (default): Fast, works offline, single-machine
- **Supabase** (when configured): Cloud storage, multi-user, scalable

## Embedding Storage

### Local (SQLite)
- Format: Binary BLOB (compact, fast)
- Performance: Fastest for <100k chunks
- Use case: Single user, local development

### Cloud (Supabase)
- Format: JSONB (query-friendly, human-readable)
- Performance: Good for large datasets
- Use case: Multi-user, cloud deployment, data sync

## Future: Hybrid Approach (Optional)

For production, consider implementing:

1. **Local-first**: Save to SQLite immediately (fast)
2. **Async sync**: Queue Supabase sync in background
3. **Smart fallback**: Query local first, cloud if needed
4. **Vector search**: Use pgvector on Supabase for similarity search

### Example Hybrid Implementation:

```python
# In document_processor.py
def save_chunk(self, chunk: Chunk):
    # Fast local save
    self.local_db.save_chunk(chunk)
    
    # Queue cloud sync (non-blocking)
    if self.cloud_available:
        self.sync_queue.add(chunk)
```

## Current Recommendation

**For now**: Use Supabase when you need cloud features (multi-device sync, backup, collaboration). Use local SQLite for development and single-user scenarios.

The auto-detection handles this automatically based on environment variables.

