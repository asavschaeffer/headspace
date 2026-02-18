use std::path::PathBuf;

/// Application configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// NVIDIA NIM API key for embeddings and multimodal extraction.
    pub nvidia_api_key: Option<String>,
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
    /// OpenRouter API key.
    pub openrouter_api_key: Option<String>,
    /// Default OpenRouter model.
    pub openrouter_model: String,
    /// vLLM OpenAI-compatible base URL.
    pub vllm_base_url: Option<String>,
    /// Default vLLM model.
    pub vllm_model: String,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// # Errors
    /// Returns an error if the data directory cannot be created.
    pub fn from_env() -> eyre::Result<Self> {
        let nvidia_api_key = read_key("NVIDIA_API_KEY");

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
                    .map(|s| s.to_ascii_lowercase())
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
            .map(|v| parse_bool(&v))
            .unwrap_or(false);
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

        Ok(Self {
            nvidia_api_key,
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
        })
    }

    /// Returns the path to the SQLite document store file.
    pub fn store_path(&self) -> PathBuf {
        self.data_dir.join("store.sqlite3")
    }

    /// Returns true if embedding generation is available.
    pub fn has_embeddings(&self) -> bool {
        self.nvidia_api_key.is_some()
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
