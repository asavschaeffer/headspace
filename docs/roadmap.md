# Headspace: Implementation Roadmap

## Current State

The system currently:
- Crawls directories recursively and extracts text from supported file types
- Generates embeddings via NVIDIA NIM API
- Runs HDBSCAN clustering and PCA projection
- Persists to SQLite (`.headspace/store.sqlite3`)
- Serves a web UI for browsing, searching, and visualizing clusters
- Supports manual re-ingestion (full re-scan each time)

What's missing:
- No file watching / incremental updates
- No human-in-the-loop review flow
- No directory oversight management (user picks which dirs to watch)
- No daemon mode

---

## Decision: Phase 1 = Ingestion Flow

**Rationale:** The daemon depends on having directories under management. Ingestion is the user-initiated "setup" phase that establishes which directories are overseen. It's cleaner to build the foundation (human review workflow) before automating maintenance.

---

## Phase 1: Ingestion (Human-in-the-Loop Review)

### Scope

**In:**
- User adds a directory to oversee → triggers ingestion flow
- For each file: AI generates assessment (description, tags, suggested path, status)
- User reviews: approve / edit / reject
- Approved files → generate embeddings → add to index
- Rejected files → marked as "rot" or ignored
- Persist: oversight list (which directories are watched)

**Out:**
- No daemon yet
- No file watching
- No auto-triggering from new files
- No re-embedding on content change

### Data Model Additions

```sql
-- Directories under oversight
CREATE TABLE overseen_dirs (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_scan_at DATETIME
);

-- File review queue
CREATE TABLE file_review (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id),
    ai_description TEXT,
    ai_tags TEXT[],        -- JSON array
    ai_suggested_path TEXT,
    ai_status TEXT,        -- rot, outdated, incomplete, complete
    user_description TEXT,
    user_tags TEXT[],
    user_suggested_path TEXT,
    user_status TEXT,
    approved BOOLEAN DEFAULT FALSE,
    reviewed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### User Flow

1. User clicks "Add Directory" → enters path
2. System crawls directory, creates Document entries with `status = 'pending_review'`
3. User navigates to "Review" tab
4. For each pending file:
   - Show AI-generated assessment (description, tags, suggested path, status)
   - User can: Approve | Edit + Approve | Reject
5. On Approve: generate embedding → add to cluster index
6. On Reject: mark as rot / excluded

### File Type Assessment

Different file types need different specialized models:
- Code files → code-specific embedding model
- PDFs → OCR + text extraction
- Images → multimodal / captioning
- Audio → transcription
- Plain text → standard embedding

This is handled by the existing `cortex/provider.rs` abstraction.

---

## Phase 2: Daemon + File Watching

### Scope

**In:**
- `notify` crate for filesystem watching
- Background task checks for changes on configurable interval
- New files → trigger review queue (notify user or auto-ingest based on preference)
- Modified files → flag for re-embedding
- Deleted files → remove from index
- System tray icon (optional)
- Auto-start on login (Windows Registry)

**Out:**
- System tray integration (Tauri later)
- Full native app

### Implementation Notes

- Use `notify` crate for cross-platform file watching
- Debounce events (don't re-trigger on every keystroke for text files)
- Store last-known content hash to detect actual changes
- Configurable watch interval (default: 60 seconds)

---

## Phase 3: Maintenance + Auto-Reorganize

### Scope

- Re-embed files when content changes significantly
- Generate migration scripts from cluster analysis
- Human-in-the-loop for applying suggestions
- Export for LLM context

---

## Phase 1 Implementation: Ingestion Flow

### Database Schema Changes

Add to `storage/mod.rs`:

```sql
-- Directories under oversight (Phase 1)
CREATE TABLE IF NOT EXISTS overseen_dirs (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at TEXT NOT NULL,
    last_scan_at TEXT
);

-- Add to files table
ALTER TABLE files ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending_review';
-- pending_review | approved | rejected

-- Add to metadata table (AI assessment, user can edit)
ALTER TABLE metadata ADD COLUMN ai_description TEXT NOT NULL DEFAULT '';
ALTER TABLE metadata ADD COLUMN ai_tags TEXT NOT NULL DEFAULT '[]';
ALTER TABLE metadata ADD COLUMN ai_suggested_path TEXT NOT NULL DEFAULT '';
ALTER TABLE metadata ADD COLUMN ai_status TEXT NOT NULL DEFAULT 'unknown';

-- User's edited versions (before final approval)
ALTER TABLE metadata ADD COLUMN user_description TEXT NOT NULL DEFAULT '';
ALTER TABLE metadata ADD COLUMN user_tags TEXT NOT NULL DEFAULT '[]';
ALTER TABLE metadata ADD COLUMN user_suggested_path TEXT NOT NULL DEFAULT '';
ALTER TABLE metadata ADD COLUMN reviewed_at TEXT;
```

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/overseen` | List overseen directories |
| `POST` | `/api/overseen` | Add directory to oversee (triggers crawl → pending) |
| `DELETE` | `/api/overseen/:id` | Remove directory from oversight |
| `GET` | `/api/review/queue` | Get files pending review |
| `GET` | `/api/review/:id` | Get single file for review |
| `POST` | `/api/review/:id` | Submit review (approve/reject + edits) |
| `POST` | `/api/review/:id/approve` | Approve (generates embedding) |
| `POST` | `/api/review/:id/reject` | Reject (mark as rot) |

### Modified Ingestion Flow

**Current behavior:**
```
crawl → extract → embed → cluster → save
```

**New behavior (Phase 1):**
```
crawl → extract → save as 'pending_review' → [wait for user] → approve → embed → cluster → update
```

The key difference: **embed step is delayed until user approval**.

### Implementation Steps

1. **Database migrations** - Add new columns and tables
2. **Storage layer** - Add methods for overseen dirs and review operations
3. **API routes** - Add endpoints for directory management and review
4. **Frontend** - Add "Oversight" tab and "Review" tab

### Frontend UI

**Oversight Tab:**
- List of overseen directories with stats (total files, pending review, approved, rejected)
- "Add Directory" button → modal with path input
- "Remove" button per directory

**Review Tab:**
- Queue of pending files (sorted by date added)
- For each file:
  - Show: filename, path, file type, AI-generated description, AI-generated tags, suggested path, AI status assessment
  - User actions: Approve | Edit & Approve | Reject
- On Approve: triggers embedding generation in background
- On Reject: marks as rejected, won't be embedded

### Key Design Decisions

1. **Files visible but not indexed**: Pending files show in UI but don't appear in search/clustering until approved
2. **AI assessment happens on crawl**: Description, tags, suggested_path generated immediately (not on-demand)
3. **User edits stored separately**: User can edit description/tags/path before approving - stored in user_* fields
4. **Re-embedding on approval**: When user approves, we generate embedding and run clustering

### Open Questions

1. **Review UI**: Tab in existing web UI, or separate lightweight view?
2. **Auto-ingest preference**: When daemon sees new file → auto-approve (if within settings) or always notify user?
3. **Daemon startup**: Registry-based auto-start (simple) or Windows Service (more robust)?
4. **LLM for descriptions**: Same NVIDIA key, or separate cheaper model?

---

## File Type Model Support

| Type | Extraction | Embedding Model |
|------|------------|-----------------|
| `.txt`, `.md`, `.json`, etc. | Direct text | Standard |
| `.rs`, `.py`, `.js`, etc. | Direct text | Code-specialized |
| `.pdf` | OCR / text extraction | Standard |
| `.png`, `.jpg`, etc. | Multimodal | Vision |
| `.mp3`, `.wav`, etc. | Transcription | Standard |

Provider abstraction already exists in `cortex/provider.rs`.
