use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Embedding dimension for NVIDIA nv-embedqa-e5-v5.
const EMBEDDING_DIM: usize = 1024;

/// Maximum input texts per batch.
const BATCH_SIZE: usize = 50;

/// NVIDIA NIM embeddings API endpoint.
const NVIDIA_EMBED_URL: &str =
    "https://integrate.api.nvidia.com/v1/embeddings";

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

/// Generates embeddings for a list of texts using NVIDIA NIM.
///
/// Returns a vector of embedding vectors in the same order as the input.
/// If no API key is configured, returns zero vectors.
///
/// # Errors
/// Returns an error if the API call fails.
pub async fn generate_embeddings(
    texts: &[String],
    config: &Config,
) -> eyre::Result<Vec<Vec<f64>>> {
    let Some(api_key) = &config.nvidia_api_key else {
        tracing::warn!("no NVIDIA API key configured; using zero vectors");
        return Ok(texts.iter().map(|_| vec![0.0; EMBEDDING_DIM]).collect());
    };

    let client = reqwest::Client::new();
    let mut all_embeddings: Vec<Vec<f64>> = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(BATCH_SIZE) {
        let truncated: Vec<String> = chunk
            .iter()
            .map(|t| {
                // Truncate to ~2000 chars to stay within token limits
                if t.len() > 2000 {
                    t[..2000].to_string()
                } else {
                    t.clone()
                }
            })
            .collect();

        let request = EmbedRequest {
            input: truncated,
            model: "nvidia/nv-embedqa-e5-v5".to_string(),
            input_type: "passage".to_string(),
            truncate: "END".to_string(),
        };

        let response = client
            .post(NVIDIA_EMBED_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(%status, body, "embedding API error");
            // Fall back to zero vectors for this batch
            for _ in chunk {
                all_embeddings.push(vec![0.0; EMBEDDING_DIM]);
            }
            continue;
        }

        let embed_response: EmbedResponse = response.json().await?;
        for obj in embed_response.data {
            all_embeddings.push(obj.embedding);
        }

        // Brief pause between batches to avoid rate limiting
        if texts.len() > BATCH_SIZE {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    Ok(all_embeddings)
}

/// Request body for NVIDIA query embeddings API.
#[derive(Serialize)]
struct QueryRequest {
    input: Vec<String>,
    model: String,
    input_type: String,
    truncate: String,
}

/// Generates an embedding for a single query string.
///
/// Uses "query" input type for better search relevance.
///
/// # Errors
/// Returns an error if the API call fails.
pub async fn generate_query_embedding(
    query: &str,
    config: &Config,
) -> eyre::Result<Vec<f64>> {
    let Some(api_key) = &config.nvidia_api_key else {
        return Ok(vec![0.0; EMBEDDING_DIM]);
    };

    let client = reqwest::Client::new();

    let request = QueryRequest {
        input: vec![query.to_string()],
        model: "nvidia/nv-embedqa-e5-v5".to_string(),
        input_type: "query".to_string(),
        truncate: "END".to_string(),
    };

    let response = client
        .post(NVIDIA_EMBED_URL)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        tracing::error!(%status, body, "query embedding API error");
        return Ok(vec![0.0; EMBEDDING_DIM]);
    }

    let embed_response: EmbedResponse = response.json().await?;
    Ok(embed_response
        .data
        .into_iter()
        .next()
        .map_or_else(|| vec![0.0; EMBEDDING_DIM], |o| o.embedding))
}
