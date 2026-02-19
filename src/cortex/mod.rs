use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::storage::Document;

pub mod provider;
pub mod router;
pub mod status;
pub mod topics;

const MAX_PREVIEW_LEN: usize = 2_000;
const MAX_SUMMARY_LEN: usize = 320;
const METADATA_VERSION: i32 = 1;

/// Pipeline family selected for a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineKind {
    Text,
    Pdf,
    Media,
    Unknown,
}

impl PipelineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Pdf => "pdf",
            Self::Media => "media",
            Self::Unknown => "unknown",
        }
    }
}

/// Input payload routed into cortex pipelines.
#[derive(Debug, Clone)]
pub struct IngestInput {
    pub abs_path: String,
    pub rel_path: String,
    pub file_id: String,
    pub bytes: Vec<u8>,
    pub mime: String,
    pub ext: String,
    pub modified_at: u64,
    pub content_hash: String,
    pub user_note: Option<String>,
}

impl IngestInput {
    /// Builds an input and detects MIME from bytes with extension fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        abs_path: String,
        rel_path: String,
        file_id: String,
        bytes: Vec<u8>,
        ext: String,
        modified_at: u64,
        content_hash: String,
    ) -> Self {
        let mime = detect_mime(&bytes, &ext);
        Self {
            abs_path,
            rel_path,
            file_id,
            bytes,
            mime,
            ext,
            modified_at,
            content_hash,
            user_note: None,
        }
    }
}

/// Output returned by a pipeline extraction stage.
#[derive(Debug, Clone)]
pub struct Extraction {
    pub pipeline_kind: PipelineKind,
    pub mime_type: String,
    pub canonical_text: String,
    pub content_preview: String,
}

/// Final metadata attached to a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadspaceMetadata {
    pub summary: String,
    pub status: String,
    pub status_confidence: f32,
    pub topics: Vec<String>,
    pub topics_confidence: f32,
    pub entities: Vec<String>,
    pub metadata_version: i32,
    pub metadata_source: String,
}

/// Summary payload returned by summarizers.
#[derive(Debug, Clone)]
pub struct SummaryOutput {
    pub summary: String,
    pub source: String,
    pub confidence: f32,
}

/// Pipeline trait for extracting canonical content.
pub trait Pipeline: Send + Sync {
    fn extract(&self, input: &IngestInput) -> eyre::Result<Extraction>;
}

/// Summarizer trait for generating concise gist text.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, text: &str) -> eyre::Result<SummaryOutput>;
}

struct TextPipeline;
struct PdfPipeline;
struct MediaPipeline;
struct UnknownPipeline;

impl Pipeline for TextPipeline {
    fn extract(&self, input: &IngestInput) -> eyre::Result<Extraction> {
        let text = String::from_utf8_lossy(&input.bytes).to_string();
        let preview = truncate_chars(&text, MAX_PREVIEW_LEN);
        Ok(Extraction {
            pipeline_kind: PipelineKind::Text,
            mime_type: input.mime.clone(),
            canonical_text: text,
            content_preview: preview,
        })
    }
}

impl Pipeline for PdfPipeline {
    fn extract(&self, input: &IngestInput) -> eyre::Result<Extraction> {
        let placeholder = format!(
            "PDF document '{}' detected. Extracting with provider fallback.",
            input.rel_path
        );
        Ok(Extraction {
            pipeline_kind: PipelineKind::Pdf,
            mime_type: input.mime.clone(),
            canonical_text: placeholder.clone(),
            content_preview: truncate_chars(&placeholder, MAX_PREVIEW_LEN),
        })
    }
}

impl Pipeline for MediaPipeline {
    fn extract(&self, input: &IngestInput) -> eyre::Result<Extraction> {
        let placeholder = format!(
            "Media file '{}' detected ({}). Extracting description with provider fallback.",
            input.rel_path, input.mime
        );
        Ok(Extraction {
            pipeline_kind: PipelineKind::Media,
            mime_type: input.mime.clone(),
            canonical_text: placeholder.clone(),
            content_preview: truncate_chars(&placeholder, MAX_PREVIEW_LEN),
        })
    }
}

impl Pipeline for UnknownPipeline {
    fn extract(&self, input: &IngestInput) -> eyre::Result<Extraction> {
        let placeholder = format!(
            "Unsupported file '{}' detected ({}).",
            input.rel_path, input.mime
        );
        Ok(Extraction {
            pipeline_kind: PipelineKind::Unknown,
            mime_type: input.mime.clone(),
            canonical_text: placeholder.clone(),
            content_preview: truncate_chars(&placeholder, MAX_PREVIEW_LEN),
        })
    }
}

struct ProviderSummarizer<'a> {
    config: &'a Config,
}

#[async_trait]
impl Summarizer for ProviderSummarizer<'_> {
    async fn summarize(&self, text: &str) -> eyre::Result<SummaryOutput> {
        let request = provider::ProviderRequest {
            task: provider::TaskKind::Summary,
            rel_path: "summary",
            mime_type: "text/plain",
            text: Some(text),
            bytes: None,
        };
        let result = provider::execute_with_fallback(&request, self.config).await?;
        Ok(SummaryOutput {
            summary: sanitize_summary(&result.text),
            source: result.provider.as_str().to_string(),
            confidence: result.confidence,
        })
    }
}

struct HeuristicSummarizer;

#[async_trait]
impl Summarizer for HeuristicSummarizer {
    async fn summarize(&self, text: &str) -> eyre::Result<SummaryOutput> {
        Ok(SummaryOutput {
            summary: heuristic_summary(text),
            source: "heuristic".to_string(),
            confidence: 0.45,
        })
    }
}

/// Enriches a document with routed extraction + generated metadata.
pub async fn enrich_document(
    document: &mut Document,
    input: &IngestInput,
    config: &Config,
) -> eyre::Result<Extraction> {
    let kind = router::select_pipeline(&input.mime, &input.ext);
    let pipeline: Box<dyn Pipeline> = match kind {
        PipelineKind::Text => Box::new(TextPipeline),
        PipelineKind::Pdf => Box::new(PdfPipeline),
        PipelineKind::Media => Box::new(MediaPipeline),
        PipelineKind::Unknown => Box::new(UnknownPipeline),
    };

    let mut extraction = pipeline.extract(input)?;
    let mut extraction_source = "heuristic".to_string();
    if let Some(result) = extract_multimodal_text(input, extraction.pipeline_kind, config).await {
        extraction.canonical_text = result.text;
        extraction.content_preview = truncate_chars(&extraction.canonical_text, MAX_PREVIEW_LEN);
        extraction_source = result.provider.as_str().to_string();
        tracing::debug!(
            file = %input.rel_path,
            provider = %result.provider.as_str(),
            model = %result.model,
            latency_ms = result.latency_ms,
            confidence = result.confidence,
            warning_count = result.warnings.len(),
            "multimodal extraction provider result"
        );
    }

    let metadata = generate_metadata(&extraction, input, document, config).await?;

    document.file_id.clone_from(&input.file_id);
    document.abs_path.clone_from(&input.abs_path);
    document.rel_path.clone_from(&input.rel_path);
    document.content_hash.clone_from(&input.content_hash);
    document.modified_at = input.modified_at;
    document.content_length = input.bytes.len();
    document.content_preview.clone_from(&extraction.content_preview);
    document.mime_type.clone_from(&extraction.mime_type);
    document.pipeline_kind = extraction.pipeline_kind.as_str().to_string();
    document.summary = metadata.summary;
    document.status = metadata.status;
    document.status_confidence = metadata.status_confidence;
    document.topics = metadata.topics;
    document.topics_confidence = metadata.topics_confidence;
    document.entities = metadata.entities;
    document.metadata_version = metadata.metadata_version;
    document.metadata_source = if metadata.metadata_source == "heuristic" {
        extraction_source
    } else {
        metadata.metadata_source
    };
    document.last_enriched_at = chrono::Utc::now().to_rfc3339();

    Ok(extraction)
}

async fn extract_multimodal_text(
    input: &IngestInput,
    kind: PipelineKind,
    config: &Config,
) -> Option<provider::ProviderResult> {
    let task = match kind {
        PipelineKind::Pdf => Some(provider::TaskKind::PdfExtract),
        PipelineKind::Media => media_task(input),
        PipelineKind::Text | PipelineKind::Unknown => None,
    }?;

    let request = provider::ProviderRequest {
        task,
        rel_path: &input.rel_path,
        mime_type: &input.mime,
        text: None,
        bytes: Some(&input.bytes),
    };

    match provider::execute_with_fallback(&request, config).await {
        Ok(result) => Some(result),
        Err(err) => {
            tracing::warn!(
                file = %input.rel_path,
                task = ?task,
                error = %err,
                "multimodal extraction fallback to deterministic placeholder"
            );
            None
        }
    }
}

fn media_task(input: &IngestInput) -> Option<provider::TaskKind> {
    if input.mime.starts_with("image/") {
        return Some(provider::TaskKind::ImageCaption);
    }
    if input.mime.starts_with("audio/") {
        return Some(provider::TaskKind::AudioTranscribe);
    }
    if input.mime.starts_with("video/") {
        return Some(provider::TaskKind::VideoDescribe);
    }

    match input.ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" | "heic" => {
            Some(provider::TaskKind::ImageCaption)
        }
        "mp3" | "wav" | "m4a" | "flac" | "ogg" => Some(provider::TaskKind::AudioTranscribe),
        "mp4" | "mov" | "mkv" | "avi" | "webm" => Some(provider::TaskKind::VideoDescribe),
        _ => None,
    }
}

fn detect_mime(bytes: &[u8], ext: &str) -> String {
    if let Some(kind) = infer::get(bytes) {
        return kind.mime_type().to_string();
    }

    if let Some(mime) = mime_guess::from_ext(ext).first_raw() {
        return mime.to_string();
    }

    if looks_like_text(bytes) {
        "text/plain".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

fn looks_like_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }

    let sample = &bytes[..bytes.len().min(1024)];
    let printable = sample
        .iter()
        .filter(|&&b| b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7E).contains(&b))
        .count();
    printable * 100 / sample.len() >= 90
}

async fn generate_metadata(
    extraction: &Extraction,
    input: &IngestInput,
    document: &Document,
    config: &Config,
) -> eyre::Result<HeadspaceMetadata> {
    let summary_prompt = if let Some(note) = input.user_note.as_deref() {
        let trimmed = note.trim();
        if trimmed.is_empty() {
            extraction.canonical_text.clone()
        } else {
            format!("User context: {trimmed}\n\n{}", extraction.canonical_text)
        }
    } else {
        extraction.canonical_text.clone()
    };
    let summary = summarize_with_fallback(&summary_prompt, config).await?;
    let (topics, topics_confidence, entities) =
        topics::extract_topics_and_entities(&extraction.canonical_text, &input.ext);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let thresholds = status::StatusThresholds {
        draft_days: config.status_draft_days,
        active_days: config.status_active_days,
        rot_days: config.status_rot_days,
        active_min_bytes: config.status_active_min_bytes,
        rot_max_bytes: config.status_rot_max_bytes,
    };
    let (status, status_confidence) = status::classify_status(
        &input.rel_path,
        &document.name,
        &input.ext,
        input.modified_at,
        input.bytes.len(),
        topics.len(),
        entities.len(),
        now,
        &thresholds,
    );

    Ok(HeadspaceMetadata {
        summary: summary.summary,
        status,
        status_confidence,
        topics,
        topics_confidence,
        entities,
        metadata_version: METADATA_VERSION,
        metadata_source: summary.source,
    })
}

async fn summarize_with_fallback(text: &str, config: &Config) -> eyre::Result<SummaryOutput> {
    let provider_summarizer = ProviderSummarizer { config };
    if let Ok(result) = provider_summarizer.summarize(text).await
        && !result.summary.trim().is_empty()
    {
        tracing::debug!(
            provider = %result.source,
            confidence = result.confidence,
            "summary generated from provider"
        );
        return Ok(result);
    }

    HeuristicSummarizer.summarize(text).await
}

fn heuristic_summary(text: &str) -> String {
    let cleaned = normalize_whitespace(text);
    if cleaned.is_empty() {
        return "No textual content could be extracted from this file.".to_string();
    }

    let mut sentences = split_sentences(&cleaned)
        .into_iter()
        .filter(|s| s.len() >= 24)
        .collect::<Vec<_>>();
    if sentences.is_empty() {
        return truncate_chars(&cleaned, MAX_SUMMARY_LEN);
    }

    if sentences.len() > 2 {
        sentences.truncate(2);
    }
    sanitize_summary(&sentences.join(" "))
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sanitize_summary(summary: &str) -> String {
    let compact = normalize_whitespace(summary);
    if compact.is_empty() {
        return compact;
    }
    truncate_chars(&compact, MAX_SUMMARY_LEN)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::heuristic_summary;

    #[test]
    fn summary_handles_empty() {
        let summary = heuristic_summary("");
        assert!(!summary.is_empty());
    }

    #[test]
    fn summary_bounds_long_input() {
        let input = "This is a long sentence about indexing and retrieval. ".repeat(120);
        let summary = heuristic_summary(&input);
        assert!(!summary.is_empty());
        assert!(summary.chars().count() <= 320);
    }

    #[test]
    fn summary_handles_short_text() {
        let summary = heuristic_summary("tiny note");
        assert!(!summary.is_empty());
    }
}
