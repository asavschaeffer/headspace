//! Vector embedding generation for semantic similarity.
//!
//! Generates embeddings via NVIDIA NIM API (or compatible OpenAI-format endpoints).
//! Falls back to zero vectors when no API key is configured.

#![allow(clippy::cast_possible_truncation, reason = "Embedding API returns f64, stored as f32")]

use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Maximum retry attempts for transient API failures (429, 5xx).
/// 3 retries provides reasonable resilience without excessive latency.
const MAX_RETRIES: u32 = 3;

/// Base delay in milliseconds for exponential backoff on retry.
/// Doubles each attempt: 500ms -> 1s -> 2s, with jitter up to +300ms.
const RETRY_BASE_MS: u64 = 500;

/// Waits with exponential backoff + jitter for a given attempt index (0-based).
async fn backoff_delay(attempt: u32) {
    use std::time::Duration;
    let base = RETRY_BASE_MS * u64::from(1u32 << attempt);
    let jitter = u64::from(rand_jitter(attempt));
    tokio::time::sleep(Duration::from_millis(base + jitter)).await;
}

/// Cheap deterministic jitter: just reuse the attempt index as a seed offset.
fn rand_jitter(attempt: u32) -> u32 {
    // Not truly random, but avoids the `rand` dependency and prevents thundering herd
    // within a single ingest run. Good enough for a single-client app.
    (attempt.wrapping_mul(6_364_136_u32)).wrapping_add(1_442_695) % 300
}

/// Embedding dimension for NVIDIA nv-embedqa-e5-v5 (and compatible models).
pub const EMBEDDING_DIM: usize = 1024;

/// Request body for NVIDIA embeddings API.
#[derive(Debug, Serialize)]
struct EmbedRequest {
    input: Vec<String>,
    model: String,
    input_type: String,
    truncate: String,
}

/// Single embedding object from the API response.
#[derive(Debug, Deserialize)]
struct EmbeddingObject {
    embedding: Vec<f64>,
}

/// API response containing embeddings.
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbeddingObject>,
}

/// Generates embeddings for a list of texts using NVIDIA NIM (or compatible endpoint).
///
/// Returns f32 vectors (downcast from the API's f64 response — no meaningful
/// precision loss for cosine similarity, and halves memory usage).
///
/// # Errors
/// Returns an error if the API call fails.
pub async fn generate_embeddings(texts: &[String], config: &Config) -> eyre::Result<Vec<Vec<f32>>> {
    let Some(api_key) = &config.embedding_api_key else {
        tracing::warn!("no embedding API key configured; using zero vectors");
        return Ok(texts.iter().map(|_| vec![0.0_f32; EMBEDDING_DIM]).collect());
    };

    let embed_url = format!("{}/embeddings", config.embedding_base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(config.embedding_batch_size) {
        let truncated: Vec<String> = chunk
            .iter()
            .map(|t| {
                let limit = config.embedding_truncate_chars;
                if t.len() > limit {
                    let end = t
                        .char_indices()
                        .map(|(i, _)| i)
                        .take_while(|&i| i <= limit)
                        .last()
                        .unwrap_or(0);
                    t[..end].to_string()
                } else {
                    t.clone()
                }
            })
            .collect();

        let request = EmbedRequest {
            input: truncated,
            model: config.embedding_model.clone(),
            input_type: "passage".to_string(),
            truncate: "END".to_string(),
        };

        let batch_result = send_batch_with_retry(
            &client,
            &embed_url,
            api_key,
            &request,
            chunk.len(),
        )
        .await;

        match batch_result {
            Ok(embeddings) => {
                all_embeddings.extend(embeddings);
            }
            Err(err) => {
                tracing::error!(error = %err, batch_size = chunk.len(), "embedding batch failed after retries; using zero vectors");
                for _ in chunk {
                    all_embeddings.push(vec![0.0_f32; EMBEDDING_DIM]);
                }
            }
        }

        // Brief pause between batches to avoid rate limiting
        if texts.len() > config.embedding_batch_size {
            tokio::time::sleep(std::time::Duration::from_millis(
                config.embedding_batch_delay_ms,
            ))
            .await;
        }
    }

    Ok(all_embeddings)
}

/// Sends a single embedding batch, retrying up to `MAX_RETRIES` times on 429.
async fn send_batch_with_retry(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    request: &EmbedRequest,
    expected_count: usize,
) -> eyre::Result<Vec<Vec<f32>>> {
    let mut last_err = String::new();
    for attempt in 0..=MAX_RETRIES {
        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            if attempt < MAX_RETRIES {
                if let Some(secs) = retry_after {
                    tracing::warn!(attempt, secs, "embedding 429 — honouring Retry-After");
                    tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                } else {
                    tracing::warn!(attempt, "embedding 429 — backing off");
                    backoff_delay(attempt).await;
                }
                last_err = format!("429 on attempt {attempt}");
                continue;
            }
            eyre::bail!("embedding rate-limited after {MAX_RETRIES} retries");
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::error!(%status, body, "embedding API error");
            eyre::bail!("http {status}");
        }

        let embed_response: EmbedResponse = response.json().await?;
        if embed_response.data.len() != expected_count {
            tracing::warn!(
                got = embed_response.data.len(),
                expected = expected_count,
                "embedding response length mismatch"
            );
        }
        return Ok(embed_response
            .data
            .into_iter()
            .map(|o| o.embedding.into_iter().map(|v| v as f32).collect())
            .collect());
    }
    eyre::bail!("embedding batch failed: {last_err}")
}

/// Generates an embedding for a single query string.
///
/// Uses "query" input type for better search relevance.
///
/// # Errors
/// Returns an error if the API call fails.
pub async fn generate_query_embedding(query: &str, config: &Config) -> eyre::Result<Vec<f32>> {
    let Some(api_key) = &config.embedding_api_key else {
        return Ok(vec![0.0_f32; EMBEDDING_DIM]);
    };

    let embed_url = format!("{}/embeddings", config.embedding_base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();

    let request = EmbedRequest {
        input: vec![query.to_string()],
        model: config.embedding_model.clone(),
        input_type: "query".to_string(),
        truncate: "END".to_string(),
    };

    let response = client
        .post(&embed_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        tracing::error!(%status, body, "query embedding API error");
        return Ok(vec![0.0_f32; EMBEDDING_DIM]);
    }

    let embed_response: EmbedResponse = response.json().await?;
    Ok(embed_response.data.into_iter().next().map_or_else(
        || vec![0.0_f32; EMBEDDING_DIM],
        |o| o.embedding.into_iter().map(|v| v as f32).collect(),
    ))
}
