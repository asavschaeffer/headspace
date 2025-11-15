# Testing Guide for Enrichment System

## Quick Test Steps

### 1. Start the Server
```bash
python headspace/main.py
```

### 2. Test Small Document (Asynchronous Enrichment)
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Small Doc",
    "content": "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.",
    "doc_type": "text"
  }'
```

**Expected**: Returns `"status": "processing"` with an `"id"`. Placeholders appear immediately in the visualization while enrichment runs in the background.

### 3. Test Large Document
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Large Doc",
    "content": "'$(python -c "print('\\n\\n'.join([f\"Paragraph {i}\" for i in range(20)]))")'",
    "doc_type": "text"
  }'
```

**Expected**: Returns `"status": "processing"` with the document ID. The websocket stream will report chunk-by-chunk updates.

### 4. Check Enrichment Status
```bash
# Replace DOC_ID with the ID from step 3
curl http://localhost:8000/api/documents/DOC_ID/status
```

**Expected**: Shows progress until enrichment finishes:
```json
{
  "document_id": "...",
  "status": "processing" | "enriched",
  "is_enriched": false | true,
  "chunks": {
    "total": 20,
    "enriched": <value>,
    "pending": <value>
  }
}
```

### 5. Test WebSocket Stream (Real-time Updates)
```javascript
const docId = "DOC_ID";
const ws = new WebSocket(`ws://localhost:8000/ws/enrichment/${docId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Enrichment event:', data);
};
```

Events:
- `started` – placeholders queued
- `chunk_enriched` – embedding + geometry morph for a chunk
- `chunk_layout_updated` – UMAP/cluster update for final positioning
- `completed` – all chunks finished

### 6. Check Server Logs
Watch for:
```
🔄 Starting enrichment for document ...
✅ Generated N embeddings
✅ Document ... enrichment complete
```

### 7. Verify Embeddings in Database
```bash
sqlite3 headspace.db "SELECT id, length(embedding) as embed_size FROM chunks WHERE document_id='DOC_ID';"
```

## Expected Behavior

### Placeholders
- Document metadata shows `status: processing`
- Chunks appear in cosmos with temporary positions/colors

### During Enrichment
- Websocket events stream chunk-by-chunk
- Shapes morph and move as events arrive
- Progress bar/status text updates in UI

### After Completion
- Document metadata switches to `status: enriched`
- Final UMAP positions and clusters are stored
- Progress reaches 100% and websocket sends `completed`

## Troubleshooting

### Issue: No WebSocket Events
- Confirm document status via `/api/documents/{id}/status`
- Check server logs for enrichment errors
- Ensure frontend includes `js/enrichment-stream.js`

### Issue: Chunks Stay at Placeholder Positions
- Ensure `chunk_layout_updated` events arrive (check console)
- Verify backend UMAP dependencies (`umap-learn`, `hdbscan`) are installed

### Issue: Planets Render Black
- Confirm home planet light and ambient lights are loading (see console warnings)
- Ensure Three.js scripts load without CDN errors

## Frontend Integration Example

```javascript
async function createThought(title, content) {
  const response = await fetch('/api/documents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, content, doc_type: 'text' })
  });
  const { id, status } = await response.json();

  await refreshCosmos(id); // fetch placeholders
  if (status === 'processing') {
    const chunkMeshes = getChunkMeshes();
    window.startEnrichmentStreaming(id, chunkMeshes);
  }
}
```

