//! Semantic similarity search using cosine distance.
//!
//! Ranks documents by cosine similarity between query and document embeddings.

use crate::storage::Document;

/// A search result with its relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    /// The matching document.
    pub document: Document,
    /// Cosine similarity score (0.0 to 1.0).
    pub score: f64,
}

/// Performs semantic search over documents using cosine similarity.
///
/// Returns the top `limit` results sorted by descending similarity.
pub fn search(query_embedding: &[f32], documents: &[&Document], limit: usize) -> Vec<SearchResult> {
    if query_embedding.is_empty() || query_embedding.iter().all(|&v| v == 0.0) {
        // No valid embedding — fall back to returning all docs
        return documents
            .iter()
            .take(limit)
            .map(|d| SearchResult {
                document: (*d).clone(),
                score: 0.0,
            })
            .collect();
    }

    let mut results: Vec<SearchResult> = documents
        .iter()
        .filter(|d| !d.embedding.is_empty() && d.embedding.iter().any(|&v| v != 0.0))
        .map(|d| {
            let score = cosine_similarity(query_embedding, &d.embedding);
            SearchResult {
                document: (*d).clone(),
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}

/// Computes cosine similarity between two f32 vectors.
///
/// Returns the result as f64 for display precision.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for i in 0..a.len() {
        let ai = f64::from(a[i]);
        let bi = f64::from(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
