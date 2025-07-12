## Introduction
The Storage Manager is the beating heart of Globule, a semantic knowledge management system that aims to transform how you capture, organize, and retrieve personal knowledge. Imagine it as a librarian who not only shelves your books but also understands their content, groups them intuitively, and fetches them instantly when you need them—all while working seamlessly across Windows, macOS, and Linux. It bridges the gap between traditional filesystems (think folders and files) and modern AI-driven capabilities like semantic search and vector embeddings. This isn’t just about storing files—it’s about making your knowledge come alive.

For this Minimum Viable Product (MVP), we’re targeting a single-user, local-first experience with 100-200 daily inputs (notes, photos, ideas), but we’re designing with an eye toward future scalability (think multiplayer collaboration). Below, I’ll unpack every layer of the design, from the nitty-gritty of database choices to the elegance of semantic directories, with plenty of examples and insights to satisfy your appetite for detail.

---

## Database Schema Design
The database is the backbone of the Storage Manager, holding everything from raw content to AI-generated embeddings. Let’s explore this in exhaustive detail.

### SQLite vs. PostgreSQL: The Great Debate
Choosing a database is like picking a car for a road trip. SQLite is a zippy little hatchback—perfect for a solo journey, easy to park, and low-maintenance. PostgreSQL is a luxury SUV—great for a group, packed with features, but overkill for a quick drive. Here’s the full breakdown:

#### **SQLite: The MVP Champion**
- **Why It Wins:**
  - **Zero Configuration:** SQLite runs in-process, meaning no server setup. You drop a `.db` file on your disk, and you’re good to go. For a single user, this is a dream—no IT degree required.
  - **Speed:** With no network overhead, SQLite delivers sub-millisecond latency for reads and writes. For 200 daily inputs, that’s snappy performance you’ll feel instantly.
  - **Portability:** One file contains everything. Move it to a USB drive, sync it with Dropbox, or email it—no fuss.
  - **Concurrency:** Write-Ahead Logging (WAL) lets multiple processes read while one writes, which is plenty for a single-user app. Think of it like a checkout line: one cashier, but lots of browsers.
  - **Proven Track Record:** Apps like Obsidian manage thousands of notes with SQLite, handling 10,000+ operations per second with the right setup.
- **The Catch:**
  - **Write Limits:** Only one writer at a time. If two processes try to write simultaneously, one waits (or fails with `SQLITE_BUSY`). For the MVP, this is fine—your app will queue writes—but it’s a red flag for future multi-user dreams.
  - **Scale Ceiling:** SQLite caps out at terabytes and billions of rows, but performance dips with heavy concurrent writes or massive datasets.

#### **PostgreSQL: The Future-Proof Contender**
- **Why It Shines:**
  - **Concurrency Superpower:** Multi-Version Concurrency Control (MVCC) lets many users read and write without stepping on each other’s toes. It’s like a multi-lane highway versus SQLite’s single track.
  - **Feature Rich:** Built-in JSONB indexing, replication, and extensions like `pgvector` for vector search make it a beast for advanced needs.
  - **Scalability:** It scales horizontally (multiple servers) and vertically (beefier hardware), ready for a “Globule Cloud” someday.
- **The Downside:**
  - **Complexity:** You need to install, configure, and maintain a server. For a local-first MVP, that’s like bringing a sledgehammer to crack a nut.
  - **Latency:** Client-server communication adds milliseconds—small, but noticeable compared to SQLite’s instant access.

#### **Verdict**
- **MVP:** SQLite, hands down. It’s lightweight, fast, and fits the single-user focus. We’ll use a Data Access Layer (DAL) to keep our options open, so switching to PostgreSQL later is just a config tweak.
- **Future:** If Globule goes multiplayer, PostgreSQL’s concurrency and features will take the wheel.

#### **Practical Example**
Imagine you jot down 10 notes in a minute. With SQLite, each write takes ~0.1ms, totaling 1ms—blink-and-you-miss-it speed. PostgreSQL might add 5-10ms of network lag, which isn’t bad, but why pay the toll for a solo trip?

---

### Embedding Storage: Vectors in the Vault
Globule uses embeddings—high-dimensional vectors (e.g., 1024 floats)—to power semantic search. Storing these efficiently is key.

#### **How We Do It**
- **BLOBs Rule:** We pack embeddings into binary large objects (BLOBs) in SQLite. A 1024-D float32 vector is 4096 bytes—compact and fast to read/write.
- **Why Not Text?** Storing as JSON (`[0.1, 0.2, ...]`) bloats the size and slows parsing. BLOBs are the lean, mean choice.
- **Vector Search:** We use `sqlite-vec`, a SQLite extension for vector indexing. It builds virtual tables (e.g., `vss_globules`) for lightning-fast K-Nearest Neighbor (KNN) queries.

#### **Code Snippet**
```sql
CREATE TABLE globules (
    id INTEGER PRIMARY KEY,
    embedding BLOB  -- 4096 bytes of binary goodness
);
CREATE VIRTUAL TABLE vss_globules USING vec0(
    item_id TEXT,
    vector FLOAT32(1024)
);
INSERT INTO vss_globules (item_id, vector) VALUES ('glob123', ?); -- Binary vector data
```

#### **Keeping It Synced**
- **Dual Tables:** The `globules` table holds the master data; `vss_globules` indexes embeddings. We tie them with `id` and use transactions to keep updates atomic:
  ```sql
  BEGIN;
  UPDATE globules SET embedding = ? WHERE id = 123;
  UPDATE vss_globules SET vector = ? WHERE item_id = '123';
  COMMIT;
  ```

#### **Edge Case**
What if an embedding update fails halfway? Transactions ensure both tables stay in sync—or roll back if something goes awry. No orphaned vectors here!

---

### JSON Field Strategies: Flexibility Meets Speed
Globules come in all flavors—notes, photos, PDFs—each with unique metadata. We need a storage strategy that’s adaptable yet performant.

#### **JSON1 Extension**
- **What It Does:** Stores metadata as JSON in TEXT columns (e.g., `{"category": "writing", "tags": ["fantasy"]}`).
- **Pros:** No schema changes needed for new fields. Add `"camera": "Nikon"` to a photo globule? Done.
- **Querying:** Use `json_extract(metadata, '$.tags')` to pluck values. It’s like a treasure hunt without rewriting the map.

#### **JSONB: The Next Level**
- **Upgrade:** SQLite 3.45.0+ offers JSONB in BLOBs—binary JSON that’s 5-10% smaller and up to 5x faster to process.
- **Trade-Off:** No O(1) lookups like PostgreSQL’s JSONB, but still a big win for frequent access.

#### **Hybrid Approach**
- **Best of Both Worlds:**
  - **Columns for Speed:** Pull out hot fields like `created_at`, `source_type`, and `tags` into dedicated columns with indexes.
  - **Generated Columns:** Index JSON fields dynamically, e.g., `ALTER TABLE globules ADD COLUMN category TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.category')) STORED;`.
  - **JSONB for the Rest:** Stash detailed stuff (e.g., photo EXIF data) in JSONB.

#### **Example Schema**
```sql
CREATE TABLE globules (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT,  -- 'note', 'photo', etc.
    metadata BLOB,     -- JSONB for {"exif": {...}, "theme": "travel"}
    embedding BLOB
);
CREATE INDEX idx_created_at ON globules(created_at);
CREATE INDEX idx_category ON globules(json_extract(metadata, '$.category'));
```

#### **Real-World Analogy**
Think of this as a filing cabinet. The labels on the drawers (`created_at`, `source_type`) are indexed for quick access. Inside, a flexible folder (JSONB) holds everything else—neat, yet adaptable.

---

### Indexing Strategies: Finding the Needle Fast
With thousands of globules, queries need to be instant. Indexes are our secret sauce.

#### **B-Tree Indexes**
- **Use Case:** Speed up `WHERE`, `JOIN`, and `ORDER BY`.
  - **Temporal:** `CREATE INDEX idx_timestamps ON globules(created_at, modified_at);`
  - **Type:** `CREATE INDEX idx_type ON globules(source_type);`

#### **Full-Text Search (FTS5)**
- **Power:** FTS5 virtual tables handle keyword searches with stemming (e.g., “run” matches “running”) and Unicode support.
- **Setup:**
  ```sql
  CREATE VIRTUAL TABLE fts_globules USING fts5(
      title, content, tags,
      content=globules,
      content_rowid=id,
      tokenize='porter unicode61'
  );
  ```
- **Query:** `SELECT * FROM fts_globules WHERE fts_globules MATCH 'semantic NEAR/5 search';`

#### **Vector Search**
- **Tool:** `sqlite-vec` indexes embeddings for similarity searches.
- **Example:** `SELECT item_id, distance FROM vss_globules WHERE vector MATCH ? ORDER BY distance LIMIT 10;`

#### **JSON Indexing**
- **Trick:** Index JSON fields with expression indexes:
  ```sql
  CREATE INDEX idx_camera ON globules(json_extract(metadata, '$.camera.model')) WHERE source_type = 'photo';
  ```

#### **Pitfall**
Too many indexes slow writes. We’ll monitor with `EXPLAIN QUERY PLAN` and prune unused ones.

---

### Full Schema Example
Here’s the whole enchilada:
```sql
CREATE TABLE globules (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT UNIQUE,
    file_size INTEGER,
    mime_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT,
    metadata BLOB,
    embedding BLOB
);

CREATE VIRTUAL TABLE vss_globules USING vec0(item_id TEXT, vector FLOAT32(1024));
CREATE VIRTUAL TABLE fts_globules USING fts5(title, content, tags, content=globules, content_rowid=id);

CREATE TABLE file_metadata (
    file_path TEXT PRIMARY KEY,
    item_id TEXT,
    last_modified TIMESTAMP,
    checksum TEXT,
    FOREIGN KEY(item_id) REFERENCES globules(id)
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
CREATE TABLE globule_tags (
    globule_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (globule_id, tag_id),
    FOREIGN KEY(globule_id) REFERENCES globules(id),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
);
```

---

## Semantic Directory Generation
Now, let’s make your filesystem as smart as your brain. Semantic directories organize files based on meaning, not just dates or manual folders.

### Path Generation: The Magic Sauce
We analyze content to build intuitive paths like `/writing/fantasy/characters`.

#### **Simple Parsed Data**
- **How:** Use Parsing Service outputs (e.g., `categories = ["writing", "fantasy"]`).
- **Example:** A note about dragons becomes `/writing/fantasy/dragons/note.md`.

#### **NLP Deep Dive**
- **Keyword Extraction:** Tools like KeyBERT pull terms (e.g., “SQLite”, “WAL” from a database note).
- **Clustering:** Hierarchical Agglomerative Clustering (HAC) builds a topic tree:
  - `Databases > Concurrency > WAL`
- **Path:** `Databases/Concurrency/WAL/optimizing-wal.md`

#### **Code Example**
```python
def generate_semantic_path(content):
    keywords = extract_keywords(content)  # ['sqlite', 'wal', 'performance']
    tree = cluster_keywords(keywords)     # { 'databases': { 'concurrency': ['wal'] } }
    path = ['databases', 'concurrency', 'wal']
    return '/'.join(path)
```

#### **Future Ideas**
- **Embeddings:** Cluster globules with k-means for tighter groups.
- **Dynamic Adjustments:** Reorganize paths as new content shifts the semantic landscape.

---

### Collision Handling
Two files can’t share the same path. Here’s how we dodge that bullet:
- **Unique Suffixes:** `dragon-lore.md` becomes `dragon-lore_123.md`.
- **Smart Naming:** Use content snippets (e.g., `dragon-lore-fire-breathing.md`).

#### **Example**
```
/writing/fantasy/dragons/dragon-lore_001.md
/writing/fantasy/dragons/dragon-lore_002.md
```

---

### Path Constraints
- **Depth:** Cap at 5 levels (`/a/b/c/d/e`) for usability.
- **Length:** Trim to 200 chars to stay under Windows’ 260-char limit (extendable with hacks).
- **Sanitization:** Replace `/` with `_`, strip `<|>`, normalize Unicode.

---

## Metadata Storage Strategies
Metadata (embeddings, tags, etc.) needs a home. We’re going hybrid.

### Options Explored
- **Extended Attributes (xattr):** Fast on Unix, but Windows says no. Sync tools often strip them too.
- **Companion Files:** `.globule` sidecars (e.g., `note.md.globule`) are portable but cluttery.
- **Database-Only:** SQLite centralizes everything, but file moves break links.

### Hybrid Solution
- **SQLite Core:** Store all metadata in the database for integrity and queries.
- **Fallback:** Embed UUIDs in file frontmatter (text files) or tiny `.globule` files (binaries).
- **Watcher:** A filesystem monitor updates paths when files move.

#### **Example Companion File**
```json
{
    "uuid": "abc123",
    "file_hash": "sha256:xyz",
    "metadata": {"tags": ["fantasy"]}
}
```

#### **Edge Case**
If you rename `note.md` to `new-note.md` outside Globule, the watcher spots it, updates `file_path` in SQLite, and all’s well.

---

## Cross-Platform Compatibility
Globule runs everywhere. Here’s how:

### Path Handling
- **Tool:** Python’s `pathlib` abstracts `/` vs `\`.
- **Storage:** Use `/` internally, translate at runtime.

### Filename Rules
- **Windows Quirks:** No `CON.txt`, cap at 255 chars per name.
- **Case:** Force lowercase (`Note.md` → `note.md`) for consistency.
- **Unicode:** Normalize to NFC (e.g., “café” stays one codepoint).

#### **Example**
`My/Note?.md` becomes `my_note_.md`.

---

## File Organization Strategies
Choose your flavor: semantic, chronological, or hybrid.

### Semantic Mode
- **Path:** `/writing/fantasy/dragons/`
- **Pro:** Matches your mental model.

### Chronological Mode
- **Path:** `/2025/07/12/note.md`
- **Pro:** Simple, unique.

### Hybrid Mode
- **Path:** `/writing/2025/07/12/note.md`
- **Pro:** Best of both.

#### **Switching**
A script re-maps paths based on your mode choice, using SQLite as the source of truth.

---

## Performance Optimization
Speed is king. Here’s the playbook:

### WAL Mode
- **Setup:** `PRAGMA journal_mode=WAL;`
- **Win:** Reads during writes, no blocking.

### Batch Writes
- **Example:** Insert 100 notes in one transaction (~50ms vs 100ms individually).

### Tuning
```sql
PRAGMA synchronous = NORMAL;  -- Faster, still safe
PRAGMA cache_size = 10000;    -- 4MB cache
PRAGMA mmap_size = 268435456; -- 256MB memory mapping
```

### Two-Stage Ingestion
- **Inbox:** Drop files in `/inbox/2025/07/12/` instantly.
- **Processor:** Background job adds embeddings, moves to semantic paths.

---

## Data Integrity and Recovery
Your data’s safe with us.

### Backups
- **Atomic:** SQLite Backup API + rsync for files.
- **Steps:**
  1. `sqlite3_backup_init` for a `.db` snapshot.
  2. Copy files with checksums.

### Recovery
- **WAL:** Rolls back crashes automatically.
- **Checks:** `PRAGMA integrity_check` spots issues.

---

## Search Implementation
Find anything, fast.

### Hybrid Search
- **Combo:** FTS5 (keywords) + sqlite-vec (vectors).
- **Ranking:** Reciprocal Rank Fusion (RRF):
  ```sql
  score = 1/(60 + fts_rank) * 0.6 + 1/(60 + vector_distance) * 0.4
  ```

#### **Example Query**
Search “dragon lore”:
- FTS5: Matches `title`, `content`.
- Vector: Finds semantically similar globules.
- RRF: Blends results for top 20 hits.
