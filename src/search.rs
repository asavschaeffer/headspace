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
pub fn search(
    query_embedding: &[f64],
    documents: &[Document],
    limit: usize,
) -> Vec<SearchResult> {
    if query_embedding.is_empty() || query_embedding.iter().all(|&v| v == 0.0) {
        // No valid embedding — fall back to returning all docs
        return documents
            .iter()
            .take(limit)
            .map(|d| SearchResult {
                document: d.clone(),
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
                document: d.clone(),
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
