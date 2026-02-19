use std::path::PathBuf;

/// Application configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// NVIDIA NIM API key for multimodal extraction and summaries.
    pub nvidia_api_key: Option<String>,
    /// Dedicated embedding API key (falls back to `nvidia_api_key` if unset).
    pub embedding_api_key: Option<String>,
    /// Embedding model name.
    pub embedding_model: String,
    /// Embedding API base URL (OpenAI-compatible, without trailing path).
    pub embedding_base_url: String,
    /// Maximum documents per embedding batch.
    pub embedding_batch_size: usize,
    /// Delay between embedding batches in milliseconds.
    pub embedding_batch_delay_ms: u64,
    /// Maximum characters to pass per document for embedding truncation.
    pub embedding_truncate_chars: usize,
    /// Port to serve the web UI on.
    pub port: u16,
    /// Directory to store headspace data.
    pub data_dir: PathBuf,
    /// Ordered provider fallback chain (cloud-first by default).
    pub provider_order: Vec<String>,
    /// Whether local inference providers are enabled.
    pub enable_local: bool,
    /// Local provider name (currently `vllm`).
    pub local_provider: String,
    /// Max file size to ingest in bytes.
    pub cortex_max_bytes: u64,
    /// Per-request timeout for provider APIs.
    pub provider_timeout_secs: u64,
    /// NVIDIA base URL (OpenAI-compatible chat completions path expected).
    pub nvidia_base_url: String,
    /// Default NVIDIA model for extraction and summaries.
    pub nvidia_model: String,
    /// Vertex chat/completions endpoint (OpenAI-compatible).
    pub vertex_chat_url: Option<String>,
    /// Vertex API key.
    pub vertex_api_key: Option<String>,
    /// Default Vertex model.
    pub vertex_model: String,
    /// `OpenRouter` API key.
    pub openrouter_api_key: Option<String>,
    /// Default `OpenRouter` model.
    pub openrouter_model: String,
    /// vLLM OpenAI-compatible base URL.
    pub vllm_base_url: Option<String>,
    /// Default vLLM model.
    pub vllm_model: String,
    /// Age threshold (days) for Draft classification (authoring file types).
    pub status_draft_days: u64,
    /// Age threshold (days) for Active classification (recent large file).
    pub status_active_days: u64,
    /// Age threshold (days) after which a small, weak-signal file may be Rot.
    pub status_rot_days: u64,
    /// Minimum content length (bytes) for Active classification.
    pub status_active_min_bytes: usize,
    /// Maximum content length (bytes) for Rot classification.
    pub status_rot_max_bytes: usize,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// # Errors
    /// Returns an error if the data directory cannot be created.
    #[allow(clippy::too_many_lines)]
    pub fn from_env() -> eyre::Result<Self> {
        let nvidia_api_key = read_key("NVIDIA_API_KEY");

        // EMBEDDING_API_KEY falls back to NVIDIA_API_KEY for backwards compatibility.
        let embedding_api_key = read_key("EMBEDDING_API_KEY").or_else(|| nvidia_api_key.clone());
        let embedding_model = read_key("EMBEDDING_MODEL")
            .unwrap_or_else(|| "nvidia/nv-embedqa-e5-v5".to_string());
        let embedding_base_url = read_key("EMBEDDING_BASE_URL")
            .unwrap_or_else(|| "https://integrate.api.nvidia.com/v1".to_string());
        let embedding_batch_size = read_key("EMBEDDING_BATCH_SIZE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        let embedding_batch_delay_ms = read_key("EMBEDDING_BATCH_DELAY_MS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let embedding_truncate_chars = read_key("EMBEDDING_TRUNCATE_CHARS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000);

        let port = read_key("PORT")
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let data_dir =
            read_key("DATA_DIR").map_or_else(|| PathBuf::from(".headspace"), PathBuf::from);
        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let provider_order = read_key("PROVIDER_ORDER")
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_ascii_lowercase)
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                vec![
                    "nvidia".to_string(),
                    "vertex".to_string(),
                    "openrouter".to_string(),
                    "vllm".to_string(),
                ]
            });

        let enable_local = read_key("ENABLE_LOCAL")
            .is_some_and(|v| parse_bool(&v));
        let local_provider = read_key("LOCAL_PROVIDER").unwrap_or_else(|| "vllm".to_string());

        let cortex_max_bytes = read_key("CORTEX_MAX_BYTES")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_048_576);
        let provider_timeout_secs = read_key("PROVIDER_TIMEOUT_SECS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        let nvidia_base_url = read_key("NVIDIA_BASE_URL")
            .unwrap_or_else(|| "https://integrate.api.nvidia.com/v1".to_string());
        let nvidia_model =
            read_key("NVIDIA_MODEL").unwrap_or_else(|| "meta/llama-3.1-70b-instruct".to_string());

        let vertex_chat_url = read_key("VERTEX_CHAT_URL").filter(|v| !v.is_empty());
        let vertex_api_key = read_key("VERTEX_API_KEY").filter(|v| !v.is_empty());
        let vertex_model =
            read_key("VERTEX_MODEL").unwrap_or_else(|| "gemini-2.5-flash".to_string());

        let openrouter_api_key = read_key("OPENROUTER_API_KEY").filter(|v| !v.is_empty());
        let openrouter_model =
            read_key("OPENROUTER_MODEL").unwrap_or_else(|| "google/gemini-2.5-flash".to_string());

        let vllm_base_url = read_key("VLLM_BASE_URL").filter(|v| !v.is_empty());
        let vllm_model = read_key("VLLM_MODEL")
            .unwrap_or_else(|| "meta-llama/Llama-3.1-8B-Instruct".to_string());

        let status_draft_days = read_key("STATUS_DRAFT_DAYS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);
        let status_active_days = read_key("STATUS_ACTIVE_DAYS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        let status_rot_days = read_key("STATUS_ROT_DAYS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(365);
        let status_active_min_bytes = read_key("STATUS_ACTIVE_MIN_BYTES")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let status_rot_max_bytes = read_key("STATUS_ROT_MAX_BYTES")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);

        Ok(Self {
            nvidia_api_key,
            embedding_api_key,
            embedding_model,
            embedding_base_url,
            embedding_batch_size,
            embedding_batch_delay_ms,
            embedding_truncate_chars,
            port,
            data_dir,
            provider_order,
            enable_local,
            local_provider,
            cortex_max_bytes,
            provider_timeout_secs,
            nvidia_base_url,
            nvidia_model,
            vertex_chat_url,
            vertex_api_key,
            vertex_model,
            openrouter_api_key,
            openrouter_model,
            vllm_base_url,
            vllm_model,
            status_draft_days,
            status_active_days,
            status_rot_days,
            status_active_min_bytes,
            status_rot_max_bytes,
        })
    }

    /// Returns the path to the `SQLite` document store file.
    pub fn store_path(&self) -> PathBuf {
        self.data_dir.join("store.sqlite3")
    }

    /// Returns true if embedding generation is available.
    pub fn has_embeddings(&self) -> bool {
        self.embedding_api_key.is_some()
    }
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn read_key(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|k| !k.is_empty())
}
