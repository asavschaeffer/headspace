# Testing Guide for Enrichment System

## Quick Test Steps

### 1. Start the Server
```bash
python headspace/main.py
```

### 2. Test Small Document (Synchronous Enrichment)
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Small Doc",
    "content": "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.",
    "doc_type": "text"
  }'
```

**Expected**: Should return `"status": "enriched"` immediately with embeddings ready.

### 3. Test Large Document (Synchronous by Default)
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Large Doc",
    "content": "'$(python -c "print('\\n\\n'.join([f\"Paragraph {i}\" for i in range(20)]))")'",
    "doc_type": "text"
  }'
```

**Expected**: Returns `"status": "enriched"` and `"id": "xxx"`. If you re-enable asynchronous enrichment, this response will revert to `"processing"`.

### 4. Check Enrichment Status
```bash
# Replace DOC_ID with the ID from step 3
curl http://localhost:8000/api/documents/DOC_ID/status
```

**Expected**: Returns enrichment summary:
```json
{
  "document_id": "...",
  "status": "enriched",
  "is_enriched": true,
  "chunks": {
    "total": 20,
    "enriched": 20,
    "pending": 0
  }
}
```

### 5. Test WebSocket Stream (Optional / Async Mode)
```javascript
// In browser console or Node.js
const ws = new WebSocket('ws://localhost:8000/ws/enrichment/DOC_ID');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Enrichment event:', data);
  
  if (data.event_type === 'chunk_enriched') {
    console.log(`Chunk ${data.chunk_index} enriched!`);
    console.log(`Progress: ${data.progress}%`);
    console.log(`Embedding dims: ${data.embedding?.length || 0}`);
  }
  
  if (data.event_type === 'completed') {
    console.log('✅ Enrichment complete!');
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

If background enrichment is disabled (default), the socket closes immediately because the document is already enriched. Re-enable asynchronous processing to see chunk-by-chunk events in real time.

### 6. Check Server Logs
Watch for these log messages:
```
🔄 Starting enrichment for document xxx
📊 Enriching 20 chunks for document xxx
✅ Generated 20 embeddings, updating chunks...
✅ Enriched 20/20 chunks for document xxx
✅ Document xxx enrichment complete
```

### 7. Verify Embeddings in Database
```bash
# For SQLite (local)
sqlite3 headspace.db "SELECT id, length(embedding) as embed_size FROM chunks WHERE document_id='DOC_ID' LIMIT 5;"

# For Supabase - check in dashboard Table Editor
```

## Expected Behavior

### Small Documents (< 10 chunks)
- ✅ Instant enrichment
- ✅ Embeddings available immediately
- ✅ 3D shapes render correctly
- ✅ No "No embedding provided" warnings

### Large Documents (> 10 chunks)
- ✅ Instant document creation
- ✅ Background enrichment starts
- ✅ WebSocket events stream in real-time
- ✅ Status endpoint shows progress
- ✅ Embeddings appear as they're generated

## Troubleshooting

### Issue: "No embedding provided" warnings
**Check**:
1. Are embeddings being generated? (check logs)
2. Is enrichment completing? (check status endpoint)
3. Are chunks being saved? (check database)

### Issue: WebSocket not receiving events
**Check**:
1. Is WebSocket connected? (`ws.readyState === 1`)
2. Is enrichment running? (check logs)
3. Is doc_id correct? (must match document ID)

### Issue: Enrichment stuck at 0%
**Check**:
1. Is embedding engine working? (check `/api/health/models`)
2. Are there errors in logs?
3. Is Supabase connection working? (if using cloud)

## Frontend Integration Example

```javascript
// Connect to WebSocket when document is created
async function createDocument(title, content) {
  const response = await fetch('/api/documents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, content, doc_type: 'text' })
  });
  
  const { id, status } = await response.json();
  
  if (status === 'processing') {
    // Connect to WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws/enrichment/${id}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.event_type === 'chunk_enriched') {
        // Update 3D visualization with new chunk
        updateChunkInVisualization(data);
      }
      
      if (data.event_type === 'completed') {
        // Refresh entire visualization
        refreshVisualization();
        ws.close();
      }
    };
  }
  
  return id;
}
```

