# Storage Manager - Low Level Design
*Version: 1.0*  
*Date: 2025-07-12*  
*Status: Draft for Review*

## 1. Introduction

This document provides the detailed low-level design for Globule's Storage Manager, the component responsible for persisting all data, generating semantic file paths, and enabling both traditional and AI-powered retrieval. The Storage Manager bridges the gap between conventional filesystem organization and modern semantic understanding, creating what we call a "semantic filesystem."

### 1.1 Scope

This LLD covers:
- SQLite database schema and optimization strategies
- Semantic path generation algorithms
- Cross-platform filesystem handling
- Transaction management and failure recovery
- Search implementation (hybrid semantic + keyword)
- Performance optimization and caching strategies

### 1.2 Dependencies from HLD

From the High Level Design document:
- Local-first architecture with single-user focus for MVP
- Support for 100-200 daily inputs (notes, photos, ideas)
- Semantic filesystem that's human-navigable
- Integration with Embedding and Parsing services
- Future scalability to multi-user scenarios

## 2. Database Architecture

### 2.1 Technology Decision: SQLite for MVP

**Decision**: SQLite is selected as the MVP database.

**Rationale**:
- Zero configuration requirement aligns with local-first philosophy
- Sub-millisecond latency for single-user workload
- Single file portability for backup and sync
- Proven track record (Obsidian handles 10,000+ ops/second)
- Write-Ahead Logging (WAL) provides adequate concurrency

**Future Migration Path**:
- Data Access Layer (DAL) abstraction allows PostgreSQL swap
- All SQL will use portable syntax where possible
- Connection pooling interface ready for client-server model

### 2.2 Core Schema Design

```sql
-- Main content table
CREATE TABLE globules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE DEFAULT (lower(hex(randomblob(16)))),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT UNIQUE,  -- SHA-256 for deduplication
    file_size INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT NOT NULL CHECK (source_type IN ('note', 'photo', 'audio', 'document')),
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    metadata BLOB,  -- JSONB format for flexibility
    embedding BLOB,  -- Binary vector data (4096 bytes for 1024-D float32)
    embedding_version INTEGER DEFAULT 1,
    embedding_updated_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_globules_created_at ON globules(created_at DESC);
CREATE INDEX idx_globules_source_type ON globules(source_type);
CREATE INDEX idx_globules_processing_status ON globules(processing_status) WHERE processing_status != 'completed';
CREATE INDEX idx_globules_file_path ON globules(file_path);
CREATE UNIQUE INDEX idx_globules_uuid ON globules(uuid);

-- Generated column for category extraction from metadata
ALTER TABLE globules ADD COLUMN category TEXT 
GENERATED ALWAYS AS (json_extract(metadata, '$.category')) STORED;
CREATE INDEX idx_globules_category ON globules(category);

-- Vector similarity search (requires sqlite-vec extension)
CREATE VIRTUAL TABLE vss_globules USING vec0(
    item_id TEXT PRIMARY KEY,
    vector FLOAT32[1024]
);

-- Full-text search
CREATE VIRTUAL TABLE fts_globules USING fts5(
    title, 
    content, 
    tags,
    content=globules,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- File system tracking
CREATE TABLE file_metadata (
    file_path TEXT PRIMARY KEY,
    globule_id INTEGER NOT NULL,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT NOT NULL,
    FOREIGN KEY(globule_id) REFERENCES globules(id) ON DELETE CASCADE
);

-- Tag management
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE globule_tags (
    globule_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (globule_id, tag_id),
    FOREIGN KEY(globule_id) REFERENCES globules(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Processing queue for async operations
CREATE TABLE processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    globule_id INTEGER NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('embed', 'parse', 'index', 'move')),
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    FOREIGN KEY(globule_id) REFERENCES globules(id) ON DELETE CASCADE
);

CREATE INDEX idx_processing_queue_priority ON processing_queue(priority DESC, created_at ASC) 
WHERE completed_at IS NULL;
```

### 2.3 Embedding Storage Strategy

Embeddings are stored as binary BLOBs for optimal performance:

```python
def store_embedding(self, globule_id: int, embedding: np.ndarray) -> None:
    """Store embedding as binary BLOB with version tracking"""
    
    # Convert numpy array to binary format
    embedding_blob = embedding.astype(np.float32).tobytes()
    
    # Update both main table and vector index atomically
    with self.db.transaction():
        self.db.execute("""
            UPDATE globules 
            SET embedding = ?, 
                embedding_version = ?,
                embedding_updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (embedding_blob, CURRENT_EMBEDDING_VERSION, globule_id))
        
        # Update vector search index
        self.db.execute("""
            INSERT OR REPLACE INTO vss_globules (item_id, vector)
            VALUES (?, ?)
        """, (str(globule_id), embedding_blob))
```

### 2.4 JSON Field Strategy

We use a hybrid approach combining JSONB with generated columns:

```python
# Example metadata structure
metadata = {
    "category": "writing",
    "subcategory": "fantasy",
    "tags": ["dragons", "worldbuilding"],
    "parsed_entities": {
        "characters": ["Aldric", "Morwen"],
        "locations": ["Dragon's Peak"]
    },
    "source_metadata": {
        # Photo EXIF, audio duration, etc.
        "camera": "Nikon D850",
        "iso": 400
    }
}

# Store as JSONB
metadata_blob = json.dumps(metadata).encode('utf-8')
```

Hot fields are extracted into generated columns for indexing, while detailed metadata remains in flexible JSONB storage.

### 2.5 Performance Optimizations

```sql
-- Enable WAL mode for concurrent reads
PRAGMA journal_mode = WAL;

-- Optimize for our workload
PRAGMA synchronous = NORMAL;  -- Faster writes, still crash-safe
PRAGMA cache_size = 10000;    -- ~40MB cache
PRAGMA temp_store = MEMORY;   -- Temp tables in RAM
PRAGMA mmap_size = 268435456; -- 256MB memory-mapped I/O

-- Analyze tables periodically for query optimization
ANALYZE;
```

## 3. Transaction Management

### 3.1 Two-Phase Transaction Pattern

We implement a two-phase pattern for robustness while maintaining performance:

```python
class TransactionManager:
    """Manages complex multi-step operations with failure recovery"""
    
    async def create_globule(self, content: str, file_path: Path) -> Globule:
        """Two-phase globule creation with compensation logic"""
        
        # Phase 1: Persistent state changes
        async with self.db.transaction() as tx:
            # 1. Write file to staging area
            staging_path = self._stage_file(content, file_path)
            
            # 2. Create database record with pending status
            globule_id = await tx.execute("""
                INSERT INTO globules (
                    title, content, file_path, file_hash, file_size,
                    mime_type, source_type, processing_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                RETURNING id
            """, (...))
            
            # 3. Move file to final location
            final_path = self._move_to_semantic_path(staging_path, globule_id)
            
            # 4. Update file path and commit
            await tx.execute("""
                UPDATE globules SET file_path = ? WHERE id = ?
            """, (str(final_path), globule_id))
            
        # Phase 2: Async processing (outside transaction)
        await self.queue_processor.enqueue([
            ProcessingTask(globule_id, 'embed', priority=5),
            ProcessingTask(globule_id, 'parse', priority=5),
            ProcessingTask(globule_id, 'index', priority=3)
        ])
        
        return await self.get_globule(globule_id)
    
    def _stage_file(self, content: str, target_path: Path) -> Path:
        """Write to temp location with atomic rename"""
        staging_dir = self.storage_root / '.staging'
        staging_path = staging_dir / f"{uuid.uuid4()}.tmp"
        
        # Write with fsync for durability
        with open(staging_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            
        return staging_path
```

### 3.2 Compensation and Recovery

```python
class RecoveryManager:
    """Handles partial failures and orphaned resources"""
    
    async def recover_on_startup(self):
        """Clean up any incomplete operations from last run"""
        
        # Find orphaned staging files
        staging_files = list((self.storage_root / '.staging').glob('*.tmp'))
        for file in staging_files:
            if file.stat().st_mtime < time.time() - 3600:  # 1 hour old
                file.unlink()
                
        # Reset stuck processing tasks
        await self.db.execute("""
            UPDATE processing_queue 
            SET started_at = NULL, retry_count = retry_count + 1
            WHERE started_at < datetime('now', '-10 minutes')
            AND completed_at IS NULL
        """)
        
        # Verify file-database consistency
        await self._verify_consistency()
    
    async def _verify_consistency(self):
        """Ensure files and database are in sync"""
        
        # Check for files without DB entries
        all_files = set(self._scan_content_files())
        db_files = set(await self._get_db_file_paths())
        
        orphaned_files = all_files - db_files
        for file_path in orphaned_files:
            await self._handle_orphaned_file(file_path)
            
        # Check for DB entries without files
        missing_files = db_files - all_files
        for file_path in missing_files:
            await self._handle_missing_file(file_path)
```

## 4. Semantic Path Generation

### 4.1 Path Generation Algorithm

The system generates intuitive paths based on content analysis:

```python
class SemanticPathGenerator:
    """Generates human-readable paths from content analysis"""
    
    def generate_path(self, 
                     parsed_data: dict, 
                     embedding: np.ndarray,
                     config: StorageConfig) -> Path:
        """Multi-strategy path generation"""
        
        if config.organization == 'semantic':
            return self._semantic_path(parsed_data, embedding)
        elif config.organization == 'chronological':
            return self._chronological_path()
        elif config.organization == 'hybrid':
            return self._hybrid_path(parsed_data)
    
    def _semantic_path(self, parsed_data: dict, embedding: np.ndarray) -> Path:
        """Generate path from semantic analysis"""
        
        # Start with parsed categories
        path_components = []
        
        # Primary category (from parsing service)
        if category := parsed_data.get('category'):
            path_components.append(self._sanitize_component(category))
            
        # Subcategory or theme
        if subcategory := parsed_data.get('subcategory'):
            path_components.append(self._sanitize_component(subcategory))
            
        # Keyword extraction for deeper hierarchy
        if len(path_components) < 3 and parsed_data.get('keywords'):
            keywords = self._extract_path_keywords(parsed_data['keywords'])
            path_components.extend(keywords[:3 - len(path_components)])
            
        # Limit depth
        path_components = path_components[:MAX_PATH_DEPTH]
        
        # Generate filename
        filename = self._generate_filename(parsed_data)
        
        return Path(*path_components) / filename
    
    def _extract_path_keywords(self, keywords: List[str]) -> List[str]:
        """Extract hierarchical keywords using NLP"""
        
        # Use KeyBERT or similar for extraction
        # Group by semantic similarity
        # Return hierarchical order
        
        # Simplified example:
        keyword_groups = self._cluster_keywords(keywords)
        return [group.representative for group in keyword_groups]
    
    def _generate_filename(self, parsed_data: dict) -> str:
        """Create descriptive filename without dates"""
        
        # Start with title or first line
        base_name = parsed_data.get('title', 'untitled')
        base_name = self._sanitize_component(base_name)
        
        # Add discriminator for uniqueness
        discriminator = parsed_data.get('key_phrase', '')
        if discriminator:
            base_name = f"{base_name}_{self._sanitize_component(discriminator)}"
            
        # Ensure uniqueness with counter if needed
        return self._ensure_unique_filename(base_name)
```

### 4.2 Collision Handling

```python
def _ensure_unique_filename(self, base_name: str, directory: Path) -> str:
    """Handle filename collisions gracefully"""
    
    # Try original name first
    if not (directory / f"{base_name}.md").exists():
        return f"{base_name}.md"
        
    # Add content-based discriminator
    for i in range(1, 100):
        candidate = f"{base_name}_{i:03d}.md"
        if not (directory / candidate).exists():
            return candidate
            
    # Fallback to UUID suffix
    return f"{base_name}_{uuid.uuid4().hex[:8]}.md"
```

### 4.3 Path Sanitization

```python
def _sanitize_component(self, component: str) -> str:
    """Make path component filesystem-safe across platforms"""
    
    # Normalize unicode to NFC
    component = unicodedata.normalize('NFC', component)
    
    # Convert to lowercase for consistency
    component = component.lower()
    
    # Replace problematic characters
    replacements = {
        '/': '_', '\\': '_', ':': '-', '*': '_',
        '?': '', '<': '', '>': '', '|': '_',
        '"': '', '\0': '', '.': '_'
    }
    
    for old, new in replacements.items():
        component = component.replace(old, new)
        
    # Strip leading/trailing dots and spaces
    component = component.strip('. ')
    
    # Limit length (leaving room for full path)
    component = component[:50]
    
    # Handle Windows reserved names
    if component.upper() in WINDOWS_RESERVED_NAMES:
        component = f"_{component}"
        
    return component or 'unnamed'
```

## 5. File System Monitoring

### 5.1 File Watcher Implementation

```python
class FileSystemMonitor:
    """Monitors filesystem for external changes"""
    
    def __init__(self, storage_root: Path, storage_manager: StorageManager):
        self.storage_root = storage_root
        self.storage_manager = storage_manager
        self.observer = Observer()  # watchdog observer
        self.pending_events = {}  # For debouncing
        
    def start(self):
        """Begin monitoring with debounced event handling"""
        
        handler = GlobuleFileHandler(self)
        self.observer.schedule(
            handler,
            str(self.storage_root),
            recursive=True
        )
        
        # Use polling observer as fallback for reliability
        if not self.observer.is_alive():
            self.observer = PollingObserver()
            self.observer.schedule(handler, str(self.storage_root), recursive=True)
            
        self.observer.start()
        
class GlobuleFileHandler(FileSystemEventHandler):
    """Handles file system events with debouncing"""
    
    def __init__(self, monitor: FileSystemMonitor):
        self.monitor = monitor
        self.debounce_delay = 0.3  # 300ms
        
    def on_moved(self, event):
        if not event.is_directory and self._is_content_file(event.dest_path):
            self._debounce_event('move', event.src_path, event.dest_path)
            
    def on_modified(self, event):
        if not event.is_directory and self._is_content_file(event.src_path):
            self._debounce_event('modify', event.src_path)
            
    def _debounce_event(self, event_type: str, *args):
        """Debounce rapid events"""
        
        key = (event_type, args[0])  # Use source path as key
        
        # Cancel existing timer
        if key in self.monitor.pending_events:
            self.monitor.pending_events[key].cancel()
            
        # Schedule new timer
        timer = threading.Timer(
            self.debounce_delay,
            self._process_event,
            args=(event_type, *args)
        )
        self.monitor.pending_events[key] = timer
        timer.start()
        
    async def _process_event(self, event_type: str, *args):
        """Process debounced event"""
        
        try:
            if event_type == 'move':
                await self._handle_move(args[0], args[1])
            elif event_type == 'modify':
                await self._handle_modify(args[0])
        except Exception as e:
            logger.error(f"Error processing {event_type} event: {e}")
```

### 5.2 Race Condition Prevention

```python
class FileLockManager:
    """Prevents concurrent access to files during processing"""
    
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()
        
    @contextmanager
    def acquire_file_lock(self, file_path: Path):
        """Acquire exclusive lock for file operations"""
        
        lock_path = file_path.with_suffix('.lock')
        
        # Try to create lock file atomically
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            
            # Write PID for debugging
            os.write(fd, str(os.getpid()).encode())
            
            try:
                yield
            finally:
                os.close(fd)
                lock_path.unlink(missing_ok=True)
                
        except FileExistsError:
            # Lock held by another process
            raise FileLockedError(f"File {file_path} is being processed")
```

## 6. Embedding Management

### 6.1 Embedding Update Strategy

```python
class EmbeddingManager:
    """Manages embedding generation and updates"""
    
    def should_regenerate_embedding(self, 
                                   old_content: str, 
                                   new_content: str,
                                   old_metadata: dict) -> bool:
        """Determine if embedding needs regeneration"""
        
        # Always regenerate if no existing embedding
        if not old_metadata.get('embedding_version'):
            return True
            
        # Check version mismatch
        if old_metadata['embedding_version'] < CURRENT_EMBEDDING_VERSION:
            return True
            
        # Check significant content change
        old_size = len(old_content)
        new_size = len(new_content)
        
        if old_size == 0:
            return True
            
        size_change_ratio = abs(new_size - old_size) / old_size
        
        # Regenerate if >20% size change
        if size_change_ratio > 0.2:
            return True
            
        # Check line count change for structured content
        old_lines = old_content.count('\n')
        new_lines = new_content.count('\n')
        
        if old_lines > 10:  # Only for substantial content
            line_change_ratio = abs(new_lines - old_lines) / old_lines
            if line_change_ratio > 0.15:
                return True
                
        # Sample content similarity (for small changes)
        if size_change_ratio < 0.05:
            # Use simple hash comparison for tiny edits
            return self._content_hash(old_content) != self._content_hash(new_content)
            
        return False
    
    async def update_embedding_batch(self, globule_ids: List[int]):
        """Batch update embeddings for efficiency"""
        
        # Fetch content in batch
        contents = await self.storage.get_contents_batch(globule_ids)
        
        # Generate embeddings in batch (more efficient)
        embeddings = await self.embedding_service.embed_batch(contents)
        
        # Update in transaction
        async with self.storage.transaction():
            for globule_id, embedding in zip(globule_ids, embeddings):
                await self.storage.store_embedding(globule_id, embedding)
```

## 7. Search Implementation

### 7.1 Hybrid Search Architecture

```python
class HybridSearchEngine:
    """Combines FTS5 keyword search with vector similarity"""
    
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.cache = SearchCache(max_size=1000, ttl=600)  # 10 min TTL
        
    async def search(self, 
                    query: str,
                    limit: int = 20,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """Perform hybrid search with RRF ranking"""
        
        # Check cache (skip if filters present)
        cache_key = self._cache_key(query, filters)
        if not filters and (cached := self.cache.get(cache_key)):
            return cached
            
        # Parallel search execution
        fts_task = self._fts_search(query, limit * 2, filters)
        vector_task = self._vector_search(query, limit * 2, filters)
        
        fts_results, vector_results = await asyncio.gather(fts_task, vector_task)
        
        # Reciprocal Rank Fusion
        combined_results = self._reciprocal_rank_fusion(
            fts_results, 
            vector_results,
            weights={'fts': 0.6, 'vector': 0.4}
        )
        
        # Apply final limit
        final_results = combined_results[:limit]
        
        # Cache if no filters
        if not filters:
            self.cache.set(cache_key, final_results)
            
        return final_results
    
    def _reciprocal_rank_fusion(self, 
                               fts_results: List[tuple],
                               vector_results: List[tuple],
                               weights: dict) -> List[SearchResult]:
        """Combine results using RRF algorithm"""
        
        k = 60  # RRF constant
        scores = {}
        
        # Process FTS results
        for rank, (id, fts_score) in enumerate(fts_results):
            rrf_score = weights['fts'] / (k + rank + 1)
            scores[id] = scores.get(id, 0) + rrf_score
            
        # Process vector results  
        for rank, (id, distance) in enumerate(vector_results):
            rrf_score = weights['vector'] / (k + rank + 1)
            scores[id] = scores.get(id, 0) + rrf_score
            
        # Sort by combined score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Fetch full records
        return self._fetch_results([id for id, _ in sorted_ids])
```

### 7.2 Search Result Caching

```python
class SearchCache:
    """LRU cache for search results with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 600):
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[List[SearchResult]]:
        """Get cached results if valid"""
        
        with self.lock:
            if key not in self.cache:
                return None
                
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
                
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key]
            
    def set(self, key: str, results: List[SearchResult]):
        """Cache search results"""
        
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]
                
            self.cache[key] = results
            self.timestamps[key] = time.time()
            
    def invalidate_all(self):
        """Clear cache on data changes"""
        
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
```

## 8. Performance Specifications

### 8.1 Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Single write (with staging) | <50ms | Includes file I/O and DB insert |
| Batch write (100 items) | <500ms | Using transaction batching |
| Semantic path generation | <10ms | Cached keyword extraction |
| FTS5 search | <20ms | For up to 10k documents |
| Vector similarity search | <50ms | Using sqlite-vec with ANN |
| Hybrid search (cached) | <5ms | LRU cache hit |
| Hybrid search (uncached) | <100ms | Combined FTS + vector + RRF |
| File move detection | <500ms | Debounced file system events |
| Embedding generation | <200ms | Via embedding service |

### 8.2 Optimization Strategies

```python
class PerformanceOptimizer:
    """System-wide performance optimizations"""
    
    async def optimize_database(self):
        """Periodic database optimization"""
        
        # Analyze tables for query planner
        await self.db.execute("ANALYZE")
        
        # Vacuum in incremental mode
        await self.db.execute("PRAGMA incremental_vacuum")
        
        # Update statistics
        await self.db.execute("""
            SELECT COUNT(*), source_type 
            FROM globules 
            GROUP BY source_type
        """)
        
    def configure_connection(self, conn):
        """Per-connection optimizations"""
        
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
```

## 9. Backup and Recovery

### 9.1 Atomic Backup Strategy

```python
class BackupManager:
    """Handles atomic backups of database and files"""
    
    async def create_backup(self, backup_path: Path) -> BackupManifest:
        """Create consistent backup of entire system"""
        
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_path / f"globule_backup_{backup_id}"
        backup_dir.mkdir(parents=True)
        
        # Phase 1: Backup SQLite database atomically
        db_backup_path = backup_dir / "globules.db"
        await self._backup_database(db_backup_path)
        
        # Phase 2: Snapshot file metadata
        file_manifest = await self._create_file_manifest()
        
        # Phase 3: Copy files with verification
        await self._backup_files(backup_dir / "files", file_manifest)
        
        # Phase 4: Create and sign manifest
        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=datetime.now(),
            db_checksum=self._checksum(db_backup_path),
            file_count=len(file_manifest),
            total_size=sum(f.size for f in file_manifest)
        )
        
        manifest.save(backup_dir / "manifest.json")
        return manifest
        
    async def _backup_database(self, target_path: Path):
        """Use SQLite backup API for consistency"""
        
        async with aiosqlite.connect(self.db_path) as source:
            async with aiosqlite.connect(target_path) as target:
                await source.backup(target)
```

## 10. Data Integrity

### 10.1 Consistency Verification

```python
class IntegrityChecker:
    """Verifies data consistency between database and filesystem"""
    
    async def verify_integrity(self) -> IntegrityReport:
        """Comprehensive integrity check"""
        
        report = IntegrityReport()
        
        # Check 1: Database integrity
        result = await self.db.execute("PRAGMA integrity_check")
        if result[0] != "ok":
            report.add_error("Database corruption detected")
            
        # Check 2: File-DB consistency
        db_files = await self._get_all_file_paths()
        fs_files = await self._scan_filesystem()
        
        # Missing files
        missing = db_files - fs_files
        for path in missing:
            report.add_warning(f"Missing file: {path}")
            
        # Orphaned files
        orphaned = fs_files - db_files
        for path in orphaned:
            report.add_warning(f"Orphaned file: {path}")
            
        # Check 3: Checksum verification (sample)
        sample_size = min(100, len(db_files))
        sample = random.sample(list(db_files), sample_size)
        
        for file_path in sample:
            stored_checksum = await self._get_stored_checksum(file_path)
            actual_checksum = await self._calculate_checksum(file_path)
            
            if stored_checksum != actual_checksum:
                report.add_error(f"Checksum mismatch: {file_path}")
                
        # Check 4: Embedding consistency
        missing_embeddings = await self.db.execute("""
            SELECT COUNT(*) FROM globules 
            WHERE embedding IS NULL 
            AND processing_status = 'completed'
        """)
        
        if missing_embeddings[0] > 0:
            report.add_warning(f"{missing_embeddings[0]} completed items missing embeddings")
            
        return report