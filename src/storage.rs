use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single ingested document with its metadata and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier.
    pub id: String,
    /// SHA-256 hash of the file's full content — the document's identity.
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
    /// Embedding vector (f32 precision, may be empty if no API key).
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
    /// When this file was last modified (epoch secs).
    pub modified_at: u64,
    /// When this document was ingested.
    pub ingested_at: String,
}

fn default_cluster() -> i32 {
    -1
}

impl Document {
    /// Creates a new document from file metadata and content.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        }
    }
}

/// The full document store persisted to disk.
///
/// Documents are keyed by their absolute path for O(1) lookups.
/// Content hashes enable change detection without re-reading files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Store {
    /// Root directory that was ingested.
    pub root_path: String,
    /// All ingested documents, keyed by absolute path.
    pub documents: HashMap<String, Document>,
}

impl Store {
    /// Load the store from a JSON file, or return a default if it doesn't exist.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be parsed.
    pub fn load(path: &std::path::Path) -> eyre::Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            let store: Self = serde_json::from_str(&data)?;
            Ok(store)
        } else {
            Ok(Self::default())
        }
    }

    /// Save the store to a JSON file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> eyre::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Find a document by its absolute path.
    pub fn find_by_path(&self, abs_path: &str) -> Option<&Document> {
        self.documents.get(abs_path)
    }

    /// Find a document by its ID (linear scan — used for API lookups).
    pub fn find_by_id(&self, id: &str) -> Option<&Document> {
        self.documents.values().find(|d| d.id == id)
    }

    /// Insert or update a document, keyed by absolute path.
    ///
    /// If a document with the same path exists:
    /// - Preserves the original `id` for URL stability
    /// - Updates all other fields
    pub fn upsert(&mut self, mut doc: Document) {
        if let Some(existing) = self.documents.get(&doc.abs_path) {
            doc.id.clone_from(&existing.id);
        }
        self.documents.insert(doc.abs_path.clone(), doc);
    }

    /// Remove documents whose absolute paths are not in `valid_paths`.
    /// Returns the number of documents removed.
    pub fn retain_existing(&mut self, valid_paths: &std::collections::HashSet<String>) -> usize {
        let before = self.documents.len();
        self.documents.retain(|path, _| valid_paths.contains(path));
        before - self.documents.len()
    }

    /// Returns all documents as a slice-like iterator (sorted by `rel_path` for deterministic output).
    pub fn documents_sorted(&self) -> Vec<&Document> {
        let mut docs: Vec<&Document> = self.documents.values().collect();
        docs.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        docs
    }
}
