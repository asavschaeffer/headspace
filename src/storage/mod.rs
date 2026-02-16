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
    /// Preview of the file content (first ~2000 chars).
    pub content_preview: String,
    /// Full text length in bytes.
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
    /// Lifecycle status marker.
    #[serde(default = "default_status")]
    pub status: String,
    /// Extracted topics.
    #[serde(default)]
    pub topics: Vec<String>,
}

fn default_cluster() -> i32 {
    -1
}

fn default_status() -> String {
    "reference".to_string()
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
            content_preview,
            content_length,
            embedding: Vec::new(),
            cluster_id: -1,
            x: 0.0,
            y: 0.0,
            modified_at,
            ingested_at: chrono::Utc::now().to_rfc3339(),
            summary: String::new(),
            status: default_status(),
            topics: Vec::new(),
        }
    }
}

/// SQLite-backed document store.
#[derive(Debug, Clone)]
pub struct Store {
    db_path: PathBuf,
    root_path: String,
}

impl Store {
    /// Opens (or creates) the SQLite store and runs schema migrations.
    pub fn load(path: &Path) -> eyre::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
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
        fetch_single(
            &conn,
            "SELECT
                f.doc_id AS id,
                f.file_id,
                f.content_hash,
                f.rel_path,
                f.abs_path,
                f.name,
                f.extension,
                f.content_preview,
                f.content_length,
                f.modified_at,
                f.ingested_at,
                m.summary,
                m.status,
                m.topics,
                m.cluster_id,
                m.x,
                m.y,
                v.embedding
             FROM files f
             LEFT JOIN metadata m ON m.file_id = f.file_id
             LEFT JOIN vectors v ON v.file_id = f.file_id
             WHERE f.abs_path = ?1
             LIMIT 1",
            params![abs_path],
        )
    }

    /// Finds a document by file identity.
    pub fn find_by_file_id(&self, file_id: &str) -> eyre::Result<Option<Document>> {
        let conn = self.connect()?;
        fetch_single(
            &conn,
            "SELECT
                f.doc_id AS id,
                f.file_id,
                f.content_hash,
                f.rel_path,
                f.abs_path,
                f.name,
                f.extension,
                f.content_preview,
                f.content_length,
                f.modified_at,
                f.ingested_at,
                m.summary,
                m.status,
                m.topics,
                m.cluster_id,
                m.x,
                m.y,
                v.embedding
             FROM files f
             LEFT JOIN metadata m ON m.file_id = f.file_id
             LEFT JOIN vectors v ON v.file_id = f.file_id
             WHERE f.file_id = ?1
             LIMIT 1",
            params![file_id],
        )
    }

    /// Finds a document by API ID.
    pub fn find_by_id(&self, id: &str) -> eyre::Result<Option<Document>> {
        let conn = self.connect()?;
        fetch_single(
            &conn,
            "SELECT
                f.doc_id AS id,
                f.file_id,
                f.content_hash,
                f.rel_path,
                f.abs_path,
                f.name,
                f.extension,
                f.content_preview,
                f.content_length,
                f.modified_at,
                f.ingested_at,
                m.summary,
                m.status,
                m.topics,
                m.cluster_id,
                m.x,
                m.y,
                v.embedding
             FROM files f
             LEFT JOIN metadata m ON m.file_id = f.file_id
             LEFT JOIN vectors v ON v.file_id = f.file_id
             WHERE f.doc_id = ?1
             LIMIT 1",
            params![id],
        )
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

    /// Returns all documents sorted by relative path.
    pub fn documents_sorted(&self) -> eyre::Result<Vec<Document>> {
        self.query_documents(
            "SELECT
                f.doc_id AS id,
                f.file_id,
                f.content_hash,
                f.rel_path,
                f.abs_path,
                f.name,
                f.extension,
                f.content_preview,
                f.content_length,
                f.modified_at,
                f.ingested_at,
                m.summary,
                m.status,
                m.topics,
                m.cluster_id,
                m.x,
                m.y,
                v.embedding
             FROM files f
             LEFT JOIN metadata m ON m.file_id = f.file_id
             LEFT JOIN vectors v ON v.file_id = f.file_id
             ORDER BY f.rel_path ASC",
        )
    }

    /// Returns all indexed documents.
    pub fn all_documents(&self) -> eyre::Result<Vec<Document>> {
        self.query_documents(
            "SELECT
                f.doc_id AS id,
                f.file_id,
                f.content_hash,
                f.rel_path,
                f.abs_path,
                f.name,
                f.extension,
                f.content_preview,
                f.content_length,
                f.modified_at,
                f.ingested_at,
                m.summary,
                m.status,
                m.topics,
                m.cluster_id,
                m.x,
                m.y,
                v.embedding
             FROM files f
             LEFT JOIN metadata m ON m.file_id = f.file_id
             LEFT JOIN vectors v ON v.file_id = f.file_id",
        )
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

    /// Flushes SQLite optimizations.
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
            content_hash TEXT NOT NULL,
            content_preview TEXT NOT NULL,
            content_length INTEGER NOT NULL,
            modified_at INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            last_seen INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (
            file_id TEXT PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            topics TEXT NOT NULL,
            cluster_id INTEGER NOT NULL DEFAULT -1,
            x REAL NOT NULL DEFAULT 0.0,
            y REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS vectors (
            file_id TEXT PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
            embedding BLOB NOT NULL
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
    let topics = serde_json::from_str(&topics_json).unwrap_or_default();

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
        content_preview: row.get("content_preview")?,
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
        topics,
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
            file_id, doc_id, abs_path, rel_path, name, extension, content_hash,
            content_preview, content_length, modified_at, ingested_at, last_seen
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, unixepoch())
         ON CONFLICT(file_id) DO UPDATE SET
            doc_id = excluded.doc_id,
            abs_path = excluded.abs_path,
            rel_path = excluded.rel_path,
            name = excluded.name,
            extension = excluded.extension,
            content_hash = excluded.content_hash,
            content_preview = excluded.content_preview,
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
            doc.content_hash,
            doc.content_preview,
            i64::try_from(doc.content_length)?,
            i64::try_from(doc.modified_at)?,
            doc.ingested_at,
        ],
    )?;

    tx.execute(
        "INSERT INTO metadata (file_id, summary, status, topics, cluster_id, x, y)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(file_id) DO UPDATE SET
            summary = excluded.summary,
            status = excluded.status,
            topics = excluded.topics,
            cluster_id = excluded.cluster_id,
            x = excluded.x,
            y = excluded.y",
        params![
            doc.file_id,
            doc.summary,
            doc.status,
            serde_json::to_string(&doc.topics)?,
            doc.cluster_id,
            doc.x,
            doc.y,
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
    let mut bytes = Vec::with_capacity(embedding.len() * std::mem::size_of::<f32>());
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
