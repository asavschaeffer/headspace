use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod file_identity;

/// A single ingested document with metadata and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stable identifier for API references.
    pub id: String,
    /// Stable file identity from OS file-id/inode.
    pub file_id: String,
    /// SHA-256 hash of the file's full content.
    pub content_hash: String,
    /// Original file path (relative to ingested root).
    pub rel_path: String,
    /// Absolute file path.
    pub abs_path: String,
    /// File name.
    pub name: String,
    /// File extension.
    pub extension: String,
    /// Human review status in the approval workflow.
    #[serde(default = "default_review_status")]
    pub review_status: String,
    /// MIME type inferred from file bytes and extension.
    #[serde(default)]
    pub mime_type: String,
    /// Routed ingestion pipeline kind.
    #[serde(default = "default_pipeline_kind")]
    pub pipeline_kind: String,
    /// Preview of the file content (first ~2000 chars).
    pub content_preview: String,
    /// Canonical extracted text used for embeddings (can exceed preview length).
    #[serde(default)]
    pub content_canonical: String,
    /// Full content length in bytes.
    pub content_length: usize,
    /// Embedding vector (f32 precision).
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// HDBSCAN cluster assignment (-1 = noise).
    #[serde(default = "default_cluster")]
    pub cluster_id: i32,
    /// 2D x coordinate for visualization.
    #[serde(default)]
    pub x: f64,
    /// 2D y coordinate for visualization.
    #[serde(default)]
    pub y: f64,
    /// Last file modification time (epoch secs).
    pub modified_at: u64,
    /// Timestamp when this document was ingested.
    pub ingested_at: String,
    /// Human-readable gist.
    #[serde(default)]
    pub summary: String,
    /// User-authored note used to steer re-enrichment.
    #[serde(default)]
    pub user_note: String,
    /// Lifecycle status marker.
    #[serde(default = "default_status")]
    pub status: String,
    /// Confidence for status classification.
    #[serde(default)]
    pub status_confidence: f32,
    /// Extracted topics.
    #[serde(default)]
    pub topics: Vec<String>,
    /// Confidence for topic extraction.
    #[serde(default)]
    pub topics_confidence: f32,
    /// Named entities discovered in content.
    #[serde(default)]
    pub entities: Vec<String>,
    /// Metadata schema version.
    #[serde(default = "default_metadata_version")]
    pub metadata_version: i32,
    /// Metadata generation source.
    #[serde(default = "default_metadata_source")]
    pub metadata_source: String,
    /// Last metadata enrichment timestamp.
    #[serde(default)]
    pub last_enriched_at: String,
    /// Indicates embedding origin: "provider", "`zero_fallback`", or "none".
    #[serde(default = "default_embedding_source")]
    pub embedding_source: String,
}

/// Overseen directory tracked for repeated scans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverseenDir {
    pub id: String,
    pub path: String,
    pub added_at: String,
    pub last_scan_at: Option<String>,
}

fn default_cluster() -> i32 {
    -1
}

fn default_status() -> String {
    "Reference".to_string()
}

fn default_review_status() -> String {
    "pending_review".to_string()
}

fn default_pipeline_kind() -> String {
    "unknown".to_string()
}

fn default_metadata_version() -> i32 {
    1
}

fn default_metadata_source() -> String {
    "heuristic".to_string()
}

fn default_embedding_source() -> String {
    "none".to_string()
}

impl Document {
    /// Creates a new document from file metadata and content.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        file_id: String,
        content_hash: String,
        rel_path: String,
        abs_path: String,
        name: String,
        extension: String,
        content_preview: String,
        content_length: usize,
        modified_at: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            file_id,
            content_hash,
            rel_path,
            abs_path,
            name,
            extension,
            review_status: default_review_status(),
            mime_type: String::new(),
            pipeline_kind: default_pipeline_kind(),
            content_preview,
            content_canonical: String::new(),
            content_length,
            embedding: Vec::new(),
            cluster_id: -1,
            x: 0.0,
            y: 0.0,
            modified_at,
            ingested_at: chrono::Utc::now().to_rfc3339(),
            summary: String::new(),
            user_note: String::new(),
            status: default_status(),
            status_confidence: 0.0,
            topics: Vec::new(),
            topics_confidence: 0.0,
            entities: Vec::new(),
            metadata_version: default_metadata_version(),
            metadata_source: default_metadata_source(),
            last_enriched_at: String::new(),
            embedding_source: default_embedding_source(),
        }
    }
}

/// SQLite-backed document store.
#[derive(Debug, Clone)]
pub struct Store {
    db_path: PathBuf,
    root_path: String,
}

const DOCUMENT_SELECT: &str = "SELECT
    f.doc_id AS id,
    f.file_id,
    f.content_hash,
    f.rel_path,
    f.abs_path,
    f.name,
    f.extension,
    f.review_status,
    f.mime_type,
    f.pipeline_kind,
    f.content_preview,
    f.content_canonical,
    f.user_note,
    f.content_length,
    f.modified_at,
    f.ingested_at,
    m.summary,
    m.status,
    m.status_confidence,
    m.topics,
    m.topics_confidence,
    m.entities,
    m.metadata_version,
    m.metadata_source,
    m.last_enriched_at,
    m.cluster_id,
    m.x,
    m.y,
    m.embedding_source,
    v.embedding
 FROM files f
 LEFT JOIN metadata m ON m.file_id = f.file_id
 LEFT JOIN vectors v ON v.file_id = f.file_id";

impl Store {
    /// Opens (or creates) the `SQLite` store and runs schema migrations.
    pub fn load(path: &Path) -> eyre::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
        }

        let conn = open_connection(path)?;
        initialize_schema(&conn)?;

        let root_path = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'root_path'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .unwrap_or_default();

        Ok(Self {
            db_path: path.to_path_buf(),
            root_path,
        })
    }

    fn connect(&self) -> eyre::Result<Connection> {
        open_connection(&self.db_path)
    }

    /// Returns current indexed root path.
    pub fn root_path(&self) -> &str {
        &self.root_path
    }

    /// Persists the indexed root path.
    pub fn set_root_path(&mut self, path: &Path) -> eyre::Result<()> {
        self.root_path = path.to_string_lossy().to_string();
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('root_path', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![self.root_path],
        )?;
        Ok(())
    }

    /// Returns number of indexed files.
    pub fn document_count(&self) -> eyre::Result<usize> {
        let conn = self.connect()?;
        let count = conn.query_row("SELECT COUNT(*) FROM files", [], |row| row.get::<_, i64>(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    /// Finds a document by absolute path.
    pub fn find_by_path(&self, abs_path: &str) -> eyre::Result<Option<Document>> {
        let conn = self.connect()?;
        let sql = format!("{DOCUMENT_SELECT} WHERE f.abs_path = ?1 LIMIT 1");
        fetch_single(&conn, &sql, params![abs_path])
    }

    /// Finds a document by file identity.
    pub fn find_by_file_id(&self, file_id: &str) -> eyre::Result<Option<Document>> {
        let conn = self.connect()?;
        let sql = format!("{DOCUMENT_SELECT} WHERE f.file_id = ?1 LIMIT 1");
        fetch_single(&conn, &sql, params![file_id])
    }

    /// Finds a document by API ID.
    pub fn find_by_id(&self, id: &str) -> eyre::Result<Option<Document>> {
        let conn = self.connect()?;
        let sql = format!("{DOCUMENT_SELECT} WHERE f.doc_id = ?1 LIMIT 1");
        fetch_single(&conn, &sql, params![id])
    }

    /// Inserts or updates a document.
    ///
    /// If a document with the same file identity exists, the API `id` is preserved.
    pub fn upsert(&mut self, mut doc: Document) -> eyre::Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;

        if let Some(existing_id) = tx
            .query_row(
                "SELECT doc_id FROM files WHERE file_id = ?1 LIMIT 1",
                params![doc.file_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            doc.id = existing_id;
        } else if let Some(existing_id) = tx
            .query_row(
                "SELECT doc_id FROM files WHERE abs_path = ?1 LIMIT 1",
                params![doc.abs_path],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            doc.id = existing_id;
        }

        write_document(&tx, &doc)?;
        tx.commit()?;
        Ok(())
    }

    /// Removes documents whose file IDs are not in `valid_file_ids`.
    pub fn retain_existing(&mut self, valid_file_ids: &HashSet<String>) -> eyre::Result<usize> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;

        let ids = {
            let mut stmt = tx.prepare("SELECT file_id FROM files")?;
            stmt.query_map([], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };

        let mut removed = 0usize;
        for file_id in ids {
            if !valid_file_ids.contains(&file_id) {
                tx.execute("DELETE FROM files WHERE file_id = ?1", params![file_id])?;
                removed += 1;
            }
        }

        tx.commit()?;
        Ok(removed)
    }

    /// Returns approved documents sorted by relative path.
    pub fn documents_sorted_approved(&self) -> eyre::Result<Vec<Document>> {
        let sql =
            format!("{DOCUMENT_SELECT} WHERE f.review_status = 'approved' ORDER BY f.rel_path ASC");
        self.query_documents(&sql)
    }

    /// Returns all approved documents.
    pub fn approved_documents(&self) -> eyre::Result<Vec<Document>> {
        let sql = format!("{DOCUMENT_SELECT} WHERE f.review_status = 'approved'");
        self.query_documents(&sql)
    }

    /// Returns all pending review documents.
    pub fn pending_review_documents(&self) -> eyre::Result<Vec<Document>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE f.review_status = 'pending_review' ORDER BY f.abs_path ASC"
        );
        self.query_documents(&sql)
    }

    /// Returns all non-rejected documents for the semantic canvas surface.
    pub fn canvas_documents(&self) -> eyre::Result<Vec<Document>> {
        let sql =
            format!("{DOCUMENT_SELECT} WHERE f.review_status != 'rejected' ORDER BY f.abs_path ASC");
        self.query_documents(&sql)
    }

    /// Sets a review status for a file identity.
    pub fn set_review_status(&mut self, file_id: &str, status: &str) -> eyre::Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "UPDATE files SET review_status = ?1, last_seen = unixepoch() WHERE file_id = ?2",
            params![status, file_id],
        )?;
        Ok(())
    }

    /// Updates the user note for a file identity.
    pub fn update_user_note(&mut self, file_id: &str, note: &str) -> eyre::Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "UPDATE files SET user_note = ?1, last_seen = unixepoch() WHERE file_id = ?2",
            params![note, file_id],
        )?;
        Ok(())
    }

    /// Writes back an updated set of documents (used after clustering).
    pub fn replace_documents(&mut self, documents: &[Document]) -> eyre::Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        for doc in documents {
            write_document(&tx, doc)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Lists all overseen directories.
    pub fn list_overseen_dirs(&self) -> eyre::Result<Vec<OverseenDir>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT id, path, added_at, last_scan_at FROM overseen_dirs ORDER BY path ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(OverseenDir {
                    id: row.get("id")?,
                    path: row.get("path")?,
                    added_at: row.get("added_at")?,
                    last_scan_at: row.get("last_scan_at")?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Adds a directory to overseen list (or returns existing if present).
    pub fn add_overseen_dir(&mut self, path: &str) -> eyre::Result<OverseenDir> {
        let conn = self.connect()?;
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO overseen_dirs (id, path, added_at, last_scan_at)
             VALUES (?1, ?2, ?3, NULL)
             ON CONFLICT(path) DO NOTHING",
            params![id, path, now],
        )?;

        let dir = conn.query_row(
            "SELECT id, path, added_at, last_scan_at FROM overseen_dirs WHERE path = ?1 LIMIT 1",
            params![path],
            |row| {
                Ok(OverseenDir {
                    id: row.get("id")?,
                    path: row.get("path")?,
                    added_at: row.get("added_at")?,
                    last_scan_at: row.get("last_scan_at")?,
                })
            },
        )?;

        Ok(dir)
    }

    /// Removes a directory from overseen list.
    pub fn remove_overseen_dir(&mut self, id: &str) -> eyre::Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM overseen_dirs WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Updates scan timestamp for an overseen directory.
    pub fn touch_overseen_dir(&mut self, id: &str) -> eyre::Result<()> {
        let conn = self.connect()?;
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE overseen_dirs SET last_scan_at = ?1 WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    /// Flushes `SQLite` optimizations.
    pub fn save(&self) -> eyre::Result<()> {
        let conn = self.connect()?;
        conn.execute_batch("PRAGMA optimize;")?;
        Ok(())
    }

    fn query_documents(&self, sql: &str) -> eyre::Result<Vec<Document>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(sql)?;
        let docs = stmt
            .query_map([], row_to_document)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(docs)
    }
}

fn open_connection(path: &Path) -> eyre::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;",
    )?;
    Ok(conn)
}

fn initialize_schema(conn: &Connection) -> eyre::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL UNIQUE,
            abs_path TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL,
            name TEXT NOT NULL,
            extension TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'pending_review',
            mime_type TEXT NOT NULL DEFAULT '',
            pipeline_kind TEXT NOT NULL DEFAULT 'unknown',
            content_hash TEXT NOT NULL,
            content_preview TEXT NOT NULL,
            content_canonical TEXT NOT NULL DEFAULT '',
            user_note TEXT NOT NULL DEFAULT '',
            content_length INTEGER NOT NULL,
            modified_at INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            last_seen INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (
            file_id TEXT PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            status_confidence REAL NOT NULL DEFAULT 0.0,
            topics TEXT NOT NULL,
            topics_confidence REAL NOT NULL DEFAULT 0.0,
            entities TEXT NOT NULL DEFAULT '[]',
            metadata_version INTEGER NOT NULL DEFAULT 1,
            metadata_source TEXT NOT NULL DEFAULT 'heuristic',
            last_enriched_at TEXT NOT NULL DEFAULT '',
            embedding_source TEXT NOT NULL DEFAULT 'none',
            cluster_id INTEGER NOT NULL DEFAULT -1,
            x REAL NOT NULL DEFAULT 0.0,
            y REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS vectors (
            file_id TEXT PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
            embedding BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS overseen_dirs (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            added_at TEXT NOT NULL,
            last_scan_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_files_abs_path ON files(abs_path);
        CREATE INDEX IF NOT EXISTS idx_files_doc_id ON files(doc_id);",
    )?;
    Ok(())
}

fn fetch_single(
    conn: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> eyre::Result<Option<Document>> {
    let mut stmt = conn.prepare(sql)?;
    let doc = stmt.query_row(params, row_to_document).optional()?;
    Ok(doc)
}

fn row_to_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
    let topics_json: String = row.get::<_, Option<String>>("topics")?.unwrap_or_default();
    let topics = serde_json::from_str::<Vec<String>>(&topics_json).unwrap_or_else(|e| {
        tracing::warn!(json = %topics_json, error = %e, "failed to parse topics JSON; defaulting to empty");
        vec![]
    });

    let entities_json: String = row
        .get::<_, Option<String>>("entities")?
        .unwrap_or_default();
    let entities = serde_json::from_str::<Vec<String>>(&entities_json).unwrap_or_else(|e| {
        tracing::warn!(json = %entities_json, error = %e, "failed to parse entities JSON; defaulting to empty");
        vec![]
    });

    let embedding_blob: Option<Vec<u8>> = row.get("embedding")?;
    let embedding = embedding_blob.map_or_else(Vec::new, |bytes| decode_embedding(&bytes));

    let content_length_i64: i64 = row.get("content_length")?;
    let modified_at_i64: i64 = row.get("modified_at")?;

    Ok(Document {
        id: row.get("id")?,
        file_id: row.get("file_id")?,
        content_hash: row.get("content_hash")?,
        rel_path: row.get("rel_path")?,
        abs_path: row.get("abs_path")?,
        name: row.get("name")?,
        extension: row.get("extension")?,
        review_status: row
            .get::<_, Option<String>>("review_status")?
            .unwrap_or_else(default_review_status),
        mime_type: row
            .get::<_, Option<String>>("mime_type")?
            .unwrap_or_default(),
        pipeline_kind: row
            .get::<_, Option<String>>("pipeline_kind")?
            .unwrap_or_else(default_pipeline_kind),
        content_preview: row.get("content_preview")?,
        content_canonical: row
            .get::<_, Option<String>>("content_canonical")?
            .unwrap_or_default(),
        user_note: row.get::<_, Option<String>>("user_note")?.unwrap_or_default(),
        content_length: usize::try_from(content_length_i64).unwrap_or(0),
        embedding,
        cluster_id: row.get::<_, Option<i32>>("cluster_id")?.unwrap_or(-1),
        x: row.get::<_, Option<f64>>("x")?.unwrap_or(0.0),
        y: row.get::<_, Option<f64>>("y")?.unwrap_or(0.0),
        modified_at: u64::try_from(modified_at_i64).unwrap_or(0),
        ingested_at: row.get("ingested_at")?,
        summary: row.get::<_, Option<String>>("summary")?.unwrap_or_default(),
        status: row
            .get::<_, Option<String>>("status")?
            .unwrap_or_else(default_status),
        status_confidence: row
            .get::<_, Option<f32>>("status_confidence")?
            .unwrap_or(0.0),
        topics,
        topics_confidence: row
            .get::<_, Option<f32>>("topics_confidence")?
            .unwrap_or(0.0),
        entities,
        metadata_version: row
            .get::<_, Option<i32>>("metadata_version")?
            .unwrap_or_else(default_metadata_version),
        metadata_source: row
            .get::<_, Option<String>>("metadata_source")?
            .unwrap_or_else(default_metadata_source),
        last_enriched_at: row
            .get::<_, Option<String>>("last_enriched_at")?
            .unwrap_or_default(),
        embedding_source: row
            .get::<_, Option<String>>("embedding_source")?
            .unwrap_or_else(default_embedding_source),
    })
}

fn write_document(tx: &Transaction<'_>, doc: &Document) -> eyre::Result<()> {
    let doc_id = if doc.id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        doc.id.clone()
    };

    // Path collisions can happen when a path is re-used by a different file identity.
    tx.execute(
        "DELETE FROM files WHERE abs_path = ?1 AND file_id != ?2",
        params![doc.abs_path, doc.file_id],
    )?;

    tx.execute(
        "INSERT INTO files (
            file_id, doc_id, abs_path, rel_path, name, extension, review_status, mime_type,
            pipeline_kind, content_hash, content_preview, content_canonical, user_note, content_length, modified_at, ingested_at, last_seen
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, unixepoch())
         ON CONFLICT(file_id) DO UPDATE SET
            doc_id = excluded.doc_id,
            abs_path = excluded.abs_path,
            rel_path = excluded.rel_path,
            name = excluded.name,
            extension = excluded.extension,
            review_status = excluded.review_status,
            mime_type = excluded.mime_type,
            pipeline_kind = excluded.pipeline_kind,
            content_hash = excluded.content_hash,
            content_preview = excluded.content_preview,
            content_canonical = excluded.content_canonical,
            user_note = excluded.user_note,
            content_length = excluded.content_length,
            modified_at = excluded.modified_at,
            ingested_at = excluded.ingested_at,
            last_seen = unixepoch()",
        params![
            doc.file_id,
            doc_id,
            doc.abs_path,
            doc.rel_path,
            doc.name,
            doc.extension,
            doc.review_status,
            doc.mime_type,
            doc.pipeline_kind,
            doc.content_hash,
            doc.content_preview,
            doc.content_canonical,
            doc.user_note,
            i64::try_from(doc.content_length)?,
            i64::try_from(doc.modified_at)?,
            doc.ingested_at,
        ],
    )?;

    tx.execute(
        "INSERT INTO metadata (
            file_id, summary, status, status_confidence, topics, topics_confidence, entities,
            metadata_version, metadata_source, last_enriched_at, cluster_id, x, y, embedding_source
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
         ON CONFLICT(file_id) DO UPDATE SET
            summary = excluded.summary,
            status = excluded.status,
            status_confidence = excluded.status_confidence,
            topics = excluded.topics,
            topics_confidence = excluded.topics_confidence,
            entities = excluded.entities,
            metadata_version = excluded.metadata_version,
            metadata_source = excluded.metadata_source,
            last_enriched_at = excluded.last_enriched_at,
            cluster_id = excluded.cluster_id,
            x = excluded.x,
            y = excluded.y,
            embedding_source = excluded.embedding_source",
        params![
            doc.file_id,
            doc.summary,
            doc.status,
            doc.status_confidence,
            serde_json::to_string(&doc.topics)?,
            doc.topics_confidence,
            serde_json::to_string(&doc.entities)?,
            doc.metadata_version,
            doc.metadata_source,
            doc.last_enriched_at,
            doc.cluster_id,
            doc.x,
            doc.y,
            doc.embedding_source,
        ],
    )?;

    tx.execute(
        "INSERT INTO vectors (file_id, embedding)
         VALUES (?1, ?2)
         ON CONFLICT(file_id) DO UPDATE SET embedding = excluded.embedding",
        params![doc.file_id, encode_embedding(&doc.embedding)],
    )?;

    Ok(())
}

fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(embedding));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
