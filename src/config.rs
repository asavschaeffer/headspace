use std::path::PathBuf;

/// Application configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// NVIDIA NIM API key for embeddings.
    pub nvidia_api_key: Option<String>,
    /// Port to serve the web UI on.
    pub port: u16,
    /// Directory to store headspace data.
    pub data_dir: PathBuf,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// # Errors
    /// Returns an error if the data directory cannot be created.
    pub fn from_env() -> eyre::Result<Self> {
        let nvidia_api_key = std::env::var("NVIDIA-API-KEY")
            .or_else(|_| std::env::var("NVIDIA_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| parse_env_file_key("NVIDIA-API-KEY"))
            .or_else(|| parse_env_file_key("NVIDIA_API_KEY"));

        let port = std::env::var("HEADSPACE_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let data_dir = std::env::var("HEADSPACE_DATA_DIR")
            .map_or_else(|_| PathBuf::from(".headspace"), PathBuf::from);

        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        Ok(Self {
            nvidia_api_key,
            port,
            data_dir,
        })
    }

    /// Returns the path to the document store file.
    pub fn store_path(&self) -> PathBuf {
        self.data_dir.join("store.json")
    }

    /// Returns true if embedding generation is available.
    pub fn has_embeddings(&self) -> bool {
        self.nvidia_api_key.is_some()
    }
}

/// Directly parses the `.env` file to find a key's value.
///
/// This is a fallback for keys with hyphens that `dotenvy` may not
/// load into the environment reliably on all platforms.
fn parse_env_file_key(key: &str) -> Option<String> {
    let content = std::fs::read_to_string(".env").ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == key {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}
