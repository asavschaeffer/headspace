use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single ingested document with its metadata and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier.
    pub id: String,
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
    /// Embedding vector (may be empty if no API key).
    #[serde(default)]
    pub embedding: Vec<f64>,
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
    pub fn new(
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Store {
    /// Root directory that was ingested.
    pub root_path: String,
    /// All ingested documents.
    pub documents: Vec<Document>,
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

    /// Find a document by its relative path.
    #[allow(dead_code)]
    pub fn find_by_path(&self, rel_path: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.rel_path == rel_path)
    }

    /// Find a document by its ID.
    pub fn find_by_id(&self, id: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.id == id)
    }
}
