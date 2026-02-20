//! File discovery and content hashing for ingestion.
//!
//! Walks directories to find candidate files, computes streaming SHA-256 hashes,
//! and collects lightweight metadata for the diff-based ingestion pipeline.

use std::collections::HashSet;
use std::io::{BufReader, Read};
use std::path::Path;

use sha2::{Digest, Sha256};
use walkdir::WalkDir;

/// Lightweight file metadata collected during discovery.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub file_id: String,
    pub content_hash: String,
    pub rel_path: String,
    pub abs_path: String,
    pub name: String,
    pub extension: String,
    pub size_bytes: u64,
    pub modified_at: u64,
}

/// Full entry with content bytes, created only when needed downstream.
#[derive(Debug, Clone)]
pub struct CrawlEntry {
    pub discovered: DiscoveredFile,
    pub bytes: Vec<u8>,
}

/// Result of scanning a directory in discovery mode.
#[derive(Debug)]
pub struct DiscoveryResult {
    /// All files found on disk.
    pub discovered: Vec<DiscoveredFile>,
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

/// Walks a directory and discovers candidate files using streaming hashes.
///
/// # Errors
/// Returns an error if the root path is not a valid directory.
pub fn discover(root: &Path, max_file_size: u64) -> eyre::Result<DiscoveryResult> {
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
        let size_bytes = metadata.len();
        if size_bytes > max_file_size {
            continue;
        }

        let Ok(content_hash) = hash_file_streaming(path) else {
            continue;
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

        discovered.push(DiscoveredFile {
            file_id,
            content_hash,
            rel_path,
            abs_path,
            name,
            extension: ext,
            size_bytes,
            modified_at,
        });
    }

    tracing::info!(count = discovered.len(), "discovery walk complete");
    Ok(DiscoveryResult {
        discovered,
        discovered_file_ids,
    })
}

fn hash_file_streaming(path: &Path) -> eyre::Result<String> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 64 * 1024];

    loop {
        let bytes = reader.read(&mut buf)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buf[..bytes]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    (name.starts_with('.') && name != ".") || SKIP_DIRS.iter().any(|&s| name == s)
}
