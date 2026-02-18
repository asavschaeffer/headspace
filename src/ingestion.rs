use std::collections::HashSet;
use std::path::Path;

use sha2::{Digest, Sha256};
use walkdir::WalkDir;

/// File metadata collected during crawl before cortex processing.
#[derive(Debug, Clone)]
pub struct CrawlEntry {
    pub file_id: String,
    pub content_hash: String,
    pub rel_path: String,
    pub abs_path: String,
    pub name: String,
    pub extension: String,
    pub bytes: Vec<u8>,
    pub modified_at: u64,
}

/// Result of crawling a directory.
#[derive(Debug)]
pub struct CrawlResult {
    /// All files found on disk.
    pub discovered: Vec<CrawlEntry>,
    /// Set of all file IDs found (for detecting deletions and renames).
    pub discovered_file_ids: HashSet<String>,
}

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

/// Extensions currently ingested by the crawler.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Text/code
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
    // Routed stubs
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "webp",
    "heic",
    "mp3",
    "wav",
    "m4a",
    "flac",
    "ogg",
    "mp4",
    "mov",
    "mkv",
    "avi",
    "webm",
];

/// Crawls a directory and returns candidate files with content hashes.
///
/// # Errors
/// Returns an error if the root path is not a valid directory.
pub fn crawl(root: &Path, max_file_size: u64) -> eyre::Result<CrawlResult> {
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

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if !SUPPORTED_EXTENSIONS.contains(&ext.as_str())
            && !SUPPORTED_EXTENSIONS.contains(&file_name.as_str())
        {
            continue;
        }

        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.len() > max_file_size {
            continue;
        }

        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };

        let content_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };

        let abs_path = path.to_string_lossy().to_string();
        let file_id = crate::storage::file_identity::file_key(path).unwrap_or_else(|e| {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "falling back to path-based identity"
            );
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

        let modified_at = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());

        discovered.push(CrawlEntry {
            file_id,
            content_hash,
            rel_path,
            abs_path,
            name,
            extension: ext,
            bytes,
            modified_at,
        });
    }

    tracing::info!(count = discovered.len(), "crawled directory");
    Ok(CrawlResult {
        discovered,
        discovered_file_ids,
    })
}

fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    (name.starts_with('.') && name != ".") || SKIP_DIRS.iter().any(|&s| name == s)
}
