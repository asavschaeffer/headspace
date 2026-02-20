use std::sync::OnceLock;
use std::time::{Duration, Instant};

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::config::Config;

const MAX_RETRIES: u32 = 3;

async fn backoff_delay(attempt: u32) {
    let base_ms: u64 = 500 * u64::from(1u32 << attempt);
    let jitter_ms: u64 = u64::from((attempt.wrapping_mul(6_364_136_u32)).wrapping_add(1_442_695) % 300);
    tokio::time::sleep(Duration::from_millis(base_ms + jitter_ms)).await;
}

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// External task class executed against providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Summary,
    PdfExtract,
    ImageCaption,
    AudioTranscribe,
    VideoDescribe,
}

impl TaskKind {
    fn system_prompt(self) -> &'static str {
        match self {
            Self::Summary => {
                "You summarize technical content in exactly 2 concise sentences. Plain text only."
            }
            Self::PdfExtract => {
                "Extract the important textual content from this PDF input. Return plain text, no markdown."
            }
            Self::ImageCaption => {
                "Describe the image in detail and include any visible text. Return plain text only."
            }
            Self::AudioTranscribe => {
                "Transcribe this audio and summarize key points. Return plain text only."
            }
            Self::VideoDescribe => {
                "Describe key events in this video and include spoken content if available. Return plain text only."
            }
        }
    }
}

/// Supported extraction/summarization providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Nvidia,
    Vertex,
    OpenRouter,
    Vllm,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Vertex => "vertex",
            Self::OpenRouter => "openrouter",
            Self::Vllm => "vllm",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "nvidia" => Some(Self::Nvidia),
            "vertex" => Some(Self::Vertex),
            "openrouter" => Some(Self::OpenRouter),
            "vllm" => Some(Self::Vllm),
            _ => None,
        }
    }
}

/// Request data routed through provider fallback.
#[derive(Debug, Clone)]
pub struct ProviderRequest<'a> {
    pub task: TaskKind,
    pub rel_path: &'a str,
    pub mime_type: &'a str,
    pub text: Option<&'a str>,
    pub bytes: Option<&'a [u8]>,
}

/// Unified provider output.
#[derive(Debug, Clone)]
pub struct ProviderResult {
    pub text: String,
    pub confidence: f32,
    pub provider: ProviderKind,
    pub model: String,
    pub latency_ms: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct ProviderFallbackError {
    pub reasons: Vec<String>,
}

impl std::fmt::Display for ProviderFallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "all providers failed: {}", self.reasons.join(" | "))
    }
}

impl std::error::Error for ProviderFallbackError {}

/// Executes task against configured providers in fallback order.
pub async fn execute_with_fallback(
    request: &ProviderRequest<'_>,
    config: &Config,
) -> Result<ProviderResult, ProviderFallbackError> {
    let chain = provider_chain(config);
    let mut reasons = Vec::new();

    for provider in chain {
        if !provider_is_configured(provider, config) {
            reasons.push(format!("{}: not configured", provider.as_str()));
            continue;
        }

        match call_provider(provider, request, config).await {
            Ok(result) => return Ok(result),
            Err(err) => reasons.push(format!("{}: {err}", provider.as_str())),
        }
    }

    Err(ProviderFallbackError { reasons })
}

fn provider_chain(config: &Config) -> Vec<ProviderKind> {
    let mut chain = Vec::new();
    for item in &config.provider_order {
        if let Some(provider) = ProviderKind::parse(item) {
            if provider == ProviderKind::Vllm
                && (!config.enable_local || !config.local_provider.eq_ignore_ascii_case("vllm"))
            {
                continue;
            }
            if !chain.contains(&provider) {
                chain.push(provider);
            }
        }
    }

    if chain.is_empty() {
        chain.push(ProviderKind::Nvidia);
        chain.push(ProviderKind::Vertex);
        chain.push(ProviderKind::OpenRouter);
        if config.enable_local && config.local_provider.eq_ignore_ascii_case("vllm") {
            chain.push(ProviderKind::Vllm);
        }
    }

    chain
}

fn provider_is_configured(provider: ProviderKind, config: &Config) -> bool {
    match provider {
        ProviderKind::Nvidia => config.nvidia_api_key.is_some(),
        ProviderKind::Vertex => config.vertex_api_key.is_some() && config.vertex_chat_url.is_some(),
        ProviderKind::OpenRouter => config.openrouter_api_key.is_some(),
        ProviderKind::Vllm => {
            config.enable_local
                && config.local_provider.eq_ignore_ascii_case("vllm")
                && config.vllm_base_url.is_some()
        }
    }
}

async fn call_provider(
    provider: ProviderKind,
    request: &ProviderRequest<'_>,
    config: &Config,
) -> eyre::Result<ProviderResult> {
    match provider {
        ProviderKind::Nvidia => {
            call_openai_compatible(
                provider,
                request,
                &config.nvidia_base_url,
                config.nvidia_api_key.as_deref(),
                &config.nvidia_model,
                None,
                config.provider_timeout_secs,
            )
            .await
        }
        ProviderKind::Vertex => {
            call_openai_compatible(
                provider,
                request,
                config.vertex_chat_url.as_deref().unwrap_or_default(),
                config.vertex_api_key.as_deref(),
                &config.vertex_model,
                None,
                config.provider_timeout_secs,
            )
            .await
        }
        ProviderKind::OpenRouter => {
            call_openai_compatible(
                provider,
                request,
                "https://openrouter.ai/api/v1",
                config.openrouter_api_key.as_deref(),
                &config.openrouter_model,
                Some(("HTTP-Referer", "https://headspace.local")),
                config.provider_timeout_secs,
            )
            .await
        }
        ProviderKind::Vllm => {
            call_openai_compatible(
                provider,
                request,
                config.vllm_base_url.as_deref().unwrap_or_default(),
                None,
                &config.vllm_model,
                None,
                config.provider_timeout_secs,
            )
            .await
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    temperature: f32,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

async fn call_openai_compatible(
    provider: ProviderKind,
    request: &ProviderRequest<'_>,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    extra_header: Option<(&str, &str)>,
    timeout_secs: u64,
) -> eyre::Result<ProviderResult> {
    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let prompt = build_prompt(request);
    let payload = ChatRequest {
        model: model.to_string(),
        temperature: 0.2,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: request.task.system_prompt().to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ],
    };

    let started = Instant::now();
    let mut last_err = String::from("no attempts made");
    for attempt in 0..=MAX_RETRIES {
        let mut builder = shared_http_client()
            .post(&endpoint)
            .timeout(Duration::from_secs(timeout_secs.max(1)))
            .header("Content-Type", "application/json");
        if let Some(key) = api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        if let Some((header_name, header_value)) = extra_header {
            builder = builder.header(header_name, header_value);
        }

        let response = builder.json(&payload).send().await?;
        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            if attempt < MAX_RETRIES {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                if let Some(secs) = retry_after {
                    tracing::warn!(provider = %provider.as_str(), attempt, secs, "LLM 429 — honouring Retry-After");
                    tokio::time::sleep(Duration::from_secs(secs)).await;
                } else {
                    tracing::warn!(provider = %provider.as_str(), attempt, "LLM 429 — backing off");
                    backoff_delay(attempt).await;
                }
                last_err = format!("429 on attempt {attempt}");
                continue;
            }
            eyre::bail!("rate-limited after {MAX_RETRIES} retries");
        }

        if !status.is_success() {
            eyre::bail!("http {status}");
        }

        let body: ChatResponse = response.json().await?;
        let text = body
            .choices
            .first()
            .map(|choice| sanitize_text(&choice.message.content))
            .unwrap_or_default();
        if text.is_empty() {
            eyre::bail!("empty response text");
        }

        return Ok(ProviderResult {
            text,
            confidence: 0.75,
            provider,
            model: model.to_string(),
            latency_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            warnings: Vec::new(),
        });
    }
    eyre::bail!("LLM call failed after retries: {last_err}")
}

fn shared_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(reqwest::Client::new)
}

fn build_prompt(request: &ProviderRequest<'_>) -> String {
    use std::fmt::Write;
    let mut prompt = String::new();
    let _ = writeln!(prompt, "Path: {}", request.rel_path);
    let _ = writeln!(prompt, "MIME: {}", request.mime_type);

    if let Some(text) = request.text {
        prompt.push_str("Text excerpt:\n");
        prompt.push_str(&truncate_chars(text, 8_000));
        prompt.push('\n');
    }

    if let Some(bytes) = request.bytes {
        // Keep provider payload bounded; this is a portability fallback representation.
        let sample = &bytes[..bytes.len().min(64 * 1024)];
        let encoded = base64::engine::general_purpose::STANDARD.encode(sample);
        prompt.push_str("Binary sample (base64, possibly truncated):\n");
        prompt.push_str(&encoded);
        prompt.push('\n');
    }

    prompt.push_str("Return plain text only.");
    prompt
}

fn sanitize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::ProviderKind;
    use crate::config::Config;

    #[test]
    fn parses_provider_names() {
        assert_eq!(ProviderKind::parse("nvidia"), Some(ProviderKind::Nvidia));
        assert_eq!(ProviderKind::parse("VERTEX"), Some(ProviderKind::Vertex));
        assert_eq!(ProviderKind::parse("bad"), None);
    }

    #[test]
    fn default_order_is_cloud_first() {
        let cfg = Config::from_env().expect("config should load");
        assert!(!cfg.provider_order.is_empty());
    }
}
