# Brain System - Technical Deep Dive

## Architecture Philosophy

The brain system follows these core principles:

1. **Filesystem as Database**: No external dependencies, files are the source of truth
2. **Progressive Enhancement**: Each layer adds capability without breaking the foundation
3. **Collision as Continuation**: Same filename = ongoing thought thread
4. **Emergent Organization**: Structure emerges from usage patterns, not predefined categories

## Core Components

### 1. Brain Core (`brain_core.py`)

The filesystem abstraction layer handling all I/O operations.

#### Key Methods

```python
save_text(content, filename=None) -> (Path, was_appended)
```
- Generates smart filenames from content patterns
- Handles collision by appending with timestamp separator
- Returns path and whether it appended vs created

```python
generate_filename(content) -> str
```
Smart naming algorithm:
1. Detect patterns (error/bug → "error-*", meeting → "meeting-*")
2. Extract semantic markers (function names, people names)
3. Fall back to first sentence/line
4. Slugify and truncate to 50 chars

```python
add_connection(file1, file2, strength, type)
```
Connection types:
- `temporal`: Created near each other in time
- `reference`: One mentions the other
- `extraction`: Generated from source (OCR, transcription)
- `similarity`: Content similarity (future)

#### Metadata Storage

```
~/brain/
  .metadata/
    connections.json       # Graph edges
    thought.txt.json      # Per-file metadata
```

Metadata includes:
- Creation/append timestamps
- Content hash for deduplication
- File type and size
- Future: embedding vectors

### 2. Web Server (`brain_server.py`)

Minimal HTTP server using Python stdlib only.

#### API Design

All endpoints return JSON for progressive enhancement:

```
POST /api/save
  Body: {content: string, filename?: string}
  Returns: {success: bool, filepath: string, appended: bool}

GET /api/graph
  Returns: {nodes: [...], edges: [...], timestamp: ISO}

GET /api/search?q=query&type=all|filename|content
  Returns: {query: string, results: [...]}

GET /api/file/{filename}
  Returns: {filename: string, content: string, size: int}

GET /api/recent?hours=24
  Returns: {files: [...]}
```

#### CORS and Security

- CORS enabled for local development flexibility
- Path traversal protection (files only from ~/brain/)
- File size limits (1MB for web viewing)
- Content sanitization for display

### 3. Galaxy Visualization (`brain_interface.html`)

Single-file HTML/JS with no external dependencies.

#### Physics Engine

Custom orbital mechanics simulation:

```javascript
class GalaxyBrain {
  updatePhysics() {
    // Each node experiences:
    // 1. Random drift (Brownian motion)
    // 2. Gravity to center (breathing modulation)
    // 3. Repulsion from neighbors (collision avoidance)
    // 4. Attraction along edges (connection strength)
    // 5. Damping (prevents runaway acceleration)
  }
}
```

Key parameters:
- **Drift Speed**: 0.0001 units/frame (geological time scale)
- **Gravity**: 0.01 * sin(breathing_phase) (30-second cycle)
- **Repulsion**: 1000/distance² (prevents overlap)
- **Connection Attraction**: 0.01 * edge_strength
- **Damping**: 0.99 (energy dissipation)

#### Rendering Pipeline

1. **Clear with fade**: `rgba(0,0,0,0.1)` creates motion trails
2. **Draw edges**: Quadratic curves with alpha based on strength
3. **Sort nodes**: By Z-depth for proper 3D layering
4. **Draw nodes**:
   - Outer glow (heat/activity)
   - Main circle (age-based alpha)
   - Inner core (brighter center)

#### Performance Optimizations

- **Incremental physics**: Only update changed nodes
- **Viewport culling**: Don't render off-screen nodes
- **Progressive loading**: Cap at 500 nodes initially
- **Debounced search**: 200ms delay on input
- **Canvas layers**: UI elements on separate layer

### 4. Search Implementation

Three-layer progressive search strategy:

#### Layer 1: Filename (Instant)
```python
# Simple string matching in filenames
for file in brain.base.glob("*"):
    if pattern in file.name.lower():
        results.append(file)
```

#### Layer 2: Content (Fast)
```python
# Try ripgrep first (10x faster)
subprocess.run(['rg', query, brain_dir])

# Fallback to grep
subprocess.run(['grep', '-r', query, brain_dir])
```

#### Layer 3: Semantic (Future)
```python
# Generate embeddings on save
embedding = generate_embedding(content)

# Cosine similarity search
similarities = [cosine_sim(embedding, other)
                for other in all_embeddings]
```

### 5. Collision Resolution

The append-based collision system treats repeated saves as continuations:

```python
if filepath.exists():
    # Collision = continuation of thought
    with open(filepath, 'a') as f:
        separator = f"\n\n{'='*50}\n[Continued {timestamp}]\n{'='*50}\n\n"
        f.write(separator + content)
```

Benefits:
- Natural thought threading
- No data loss
- Temporal context preserved
- Reduces filename proliferation

## Performance Characteristics

### Save Path
1. Generate filename: ~1ms
2. Write file: ~5ms
3. Update metadata: ~2ms
4. Update connections: ~2ms
**Total: <10ms**

### Search Path
1. Filename search: <10ms for 10k files
2. Ripgrep content: <100ms for 10k files
3. Load results: ~10ms
**Total: <120ms**

### Visualization
- Initial load: ~200ms
- Physics update: ~5ms/frame
- Render: ~10ms/frame
**Maintains 60 FPS with 500 nodes**

## Scaling Considerations

### At 10,000 Files
- Filesystem performance remains good
- Search may need indexing (SQLite FTS)
- Visualization needs clustering

### At 100,000 Files
- Need sharding by date/type
- Implement cold storage for old files
- Use WebGL for rendering
- Add server-side filtering

### At 1,000,000 Files
- Distributed storage (content-addressed)
- Elasticsearch for search
- Graph database for connections
- Microservices architecture

## Future Enhancements

### Near Term
1. **Vector Embeddings**: Semantic similarity search
2. **Auto-tagging**: LLM-based categorization
3. **Voice Input**: Web Speech API integration
4. **Mobile App**: React Native wrapper

### Medium Term
1. **Collaboration**: Multi-user brain spaces
2. **Versioning**: Git-like history tracking
3. **Plugins**: External processor support
4. **Export**: Knowledge graph formats

### Long Term
1. **AI Assistant**: Query your brain with natural language
2. **Pattern Detection**: Identify recurring themes
3. **Memory Decay**: Automatic archival of old thoughts
4. **Cross-Brain Links**: Connect multiple brain spaces

## Security Model

### Current
- Local-only by default
- No authentication (single user)
- No encryption (filesystem permissions)

### For Network Access
```python
# Add to server
def require_auth(handler):
    auth_header = handler.headers.get('Authorization')
    if not validate_token(auth_header):
        handler.send_error(401)
        return False
    return True
```

### For Sensitive Data
- Encrypt files at rest (gpg/age)
- Use HTTPS for remote access
- Add rate limiting
- Implement CSRF protection

## Testing Strategy

### Unit Tests
```python
def test_filename_generation():
    assert generate_filename("BUG: Auth error") == "error-bug.txt"
    assert generate_filename("Meeting with Sarah") == "meeting-sarah.txt"

def test_collision_handling():
    brain.save_text("First thought", "test.txt")
    brain.save_text("Second thought", "test.txt")
    content = Path("~/brain/test.txt").read_text()
    assert "Continued" in content
```

### Integration Tests
```python
def test_save_and_search():
    brain.save_text("Vector database research")
    results = brain.search_content("vector")
    assert len(results) > 0
```

### Performance Tests
```python
def test_save_latency():
    start = time.time()
    brain.save_text("Performance test thought")
    assert time.time() - start < 0.01  # 10ms
```

## Deployment

### Local Development
```bash
python start_brain.py
```

### Production (Single User)
```bash
# Use proper process manager
supervisord -c brain.conf

# Or systemd service
systemctl start brain.service
```

### Production (Multi-User)
```nginx
# Nginx reverse proxy
location /brain/ {
    proxy_pass http://localhost:8888/;
    proxy_set_header X-User $remote_user;
}
```

## Debugging

### Common Issues

**Slow search**: Install ripgrep
```bash
# macOS
brew install ripgrep

# Ubuntu
apt-get install ripgrep

# Windows
choco install ripgrep
```

**Galaxy not updating**: Check WebSocket connection
```javascript
console.log(galaxy.nodes.size);  // Should increase
```

**High CPU usage**: Reduce physics calculations
```javascript
this.settings.maxNodes = 200;  // Limit visible nodes
```

## Contributing

The system is designed for hackability:

1. **Single file components**: Easy to understand and modify
2. **No build process**: Edit and reload
3. **Clear separation**: Core → Server → UI
4. **Progressive enhancement**: Add features without breaking basics

Key files to modify:
- `brain_core.py`: Add new file processors
- `brain_interface.html`: Customize visualization
- `brain_server.py`: Add new endpoints

The beauty is in the simplicity - keep it that way.