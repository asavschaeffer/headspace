use std::collections::HashSet;
use std::path::Path;

use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::storage::Document;

/// File extensions we can extract text from.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "rs",
    "py",
    "js",
    "ts",
    "jsx",
    "tsx",
    "json",
    "toml",
    "yaml",
    "yml",
    "html",
    "css",
    "scss",
    "c",
    "cpp",
    "h",
    "hpp",
    "go",
    "java",
    "sh",
    "bat",
    "ps1",
    "xml",
    "csv",
    "log",
    "cfg",
    "ini",
    "env",
    "sql",
    "rb",
    "php",
    "swift",
    "kt",
    "r",
    "lua",
    "pl",
    "ex",
    "exs",
    "zig",
    "nim",
    "v",
    "d",
    "makefile",
    "dockerfile",
];

/// Directories to skip during crawl.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".headspace",
    "dist",
    "build",
    ".next",
];

/// Maximum file size to ingest (1 MB).
const MAX_FILE_SIZE: u64 = 1_048_576;

/// Maximum preview length in characters.
const MAX_PREVIEW_LEN: usize = 2000;

/// Result of crawling a directory.
#[derive(Debug)]
pub struct CrawlResult {
    /// All files found on disk (with content hashes computed).
    pub discovered: Vec<Document>,
    /// Set of all file IDs found (for detecting deletions and renames).
    pub discovered_file_ids: HashSet<String>,
}

/// Crawls a directory and returns discovered documents with content hashes.
///
/// Each document has its SHA-256 content hash computed during crawl,
/// enabling efficient change detection without re-reading files later.
///
/// # Errors
/// Returns an error if the root path is not a valid directory.
pub fn crawl(root: &Path) -> eyre::Result<CrawlResult> {
    if !root.is_dir() {
        eyre::bail!("path is not a directory: {}", root.display());
    }

    let root = root.canonicalize()?;
    let mut discovered = Vec::new();
    let mut discovered_file_ids = HashSet::new();

    for entry in WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_skipped_dir(e))
    {
        let Ok(entry) = entry else {
            continue;
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        // Check extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Also handle extensionless files like Makefile, Dockerfile
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !SUPPORTED_EXTENSIONS.contains(&ext.as_str())
            && !SUPPORTED_EXTENSIONS.contains(&file_name.as_str())
        {
            continue;
        }

        // Check file size
        let Ok(metadata) = entry.metadata() else {
            continue;
        };

        if metadata.len() > MAX_FILE_SIZE {
            continue;
        }

        // Read content
        let Ok(content) = std::fs::read_to_string(path) else {
            continue; // Skip binary / unreadable files
        };

        // Compute SHA-256 content hash
        let content_hash = {
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let abs_path = path.to_string_lossy().to_string();
        let file_id =
            crate::storage::file_identity::file_key(path).unwrap_or_else(|e| {
                tracing::warn!(path = %path.display(), error = %e, "falling back to path-based identity");
                crate::storage::file_identity::fallback_file_key(path)
            });
        discovered_file_ids.insert(file_id.clone());

        let rel_path = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let content_length = content.len();
        let content_preview = if content.len() > MAX_PREVIEW_LEN {
            // Find a safe char boundary to truncate at
            let end = content
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= MAX_PREVIEW_LEN)
                .last()
                .unwrap_or(0);
            content[..end].to_string()
        } else {
            content
        };

        let modified_at = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());

        discovered.push(Document::new(
            file_id,
            content_hash,
            rel_path,
            abs_path,
            name,
            ext,
            content_preview,
            content_length,
            modified_at,
        ));
    }

    tracing::info!(count = discovered.len(), "crawled directory");
    Ok(CrawlResult {
        discovered,
        discovered_file_ids,
    })
}

/// Returns true if a directory entry should be skipped.
fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    // Skip hidden directories (starting with '.') and known skip dirs
    (name.starts_with('.') && name != ".") || SKIP_DIRS.iter().any(|&s| name == s)
}
