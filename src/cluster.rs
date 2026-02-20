//! Document clustering and 2D visualization projection.
//!
//! Uses HDBSCAN for density-based clustering and t-SNE/PCA for dimensionality
//! reduction to produce 2D coordinates for the semantic canvas.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "Clustering math requires f64/f32 conversions for ndarray/hdbscan interop"
)]

use hdbscan::Hdbscan;
use linfa::traits::Transformer as _;
use linfa_tsne::TSneParams;
use ndarray::Array2;

use crate::storage::Document;

/// Runs HDBSCAN clustering on document embeddings and assigns 2D coordinates.
///
/// Updates each document's `cluster_id`, `x`, and `y` fields in place.
pub fn cluster_documents(documents: &mut [Document]) {
    if documents.len() < 3 {
        // Not enough documents to cluster — assign 2D positions only
        assign_2d_positions(documents);
        return;
    }

    // Check if any documents have valid embeddings
    let has_embeddings = documents
        .iter()
        .any(|d| !d.embedding.is_empty() && d.embedding.iter().any(|&v| v != 0.0));

    if !has_embeddings {
        assign_2d_positions(documents);
        return;
    }

    // Build embedding matrix — hdbscan requires Vec<Vec<f64>>, so upcast at the boundary
    let data: Vec<Vec<f64>> = documents
        .iter()
        .map(|d| {
            if d.embedding.is_empty() {
                vec![0.0; crate::embeddings::EMBEDDING_DIM]
            } else {
                d.embedding.iter().map(|&v| f64::from(v)).collect()
            }
        })
        .collect();

    // Run HDBSCAN
    let clusterer = Hdbscan::default_hyper_params(&data);

    match clusterer.cluster() {
        Ok(labels) => {
            let num_clusters = labels
                .iter()
                .filter(|&&l| l >= 0)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let noise_count = labels.iter().filter(|&&l| l < 0).count();

            for (i, &label) in labels.iter().enumerate() {
                documents[i].cluster_id = label;
            }

            tracing::info!(
                clusters = num_clusters,
                noise = noise_count,
                "HDBSCAN clustering complete"
            );
        }
        Err(e) => {
            tracing::warn!("HDBSCAN clustering failed: {e:?}");
        }
    }

    // Assign 2D positions via PCA projection
    assign_2d_positions(documents);
}

/// Minimum number of documents for t-SNE to produce meaningful results.
/// t-SNE requires perplexity < n/3, and perplexity >= 5 is recommended.
/// 10 samples allows perplexity of 3 with some margin.
const TSNE_MIN_SAMPLES: usize = 10;

/// Maximum t-SNE perplexity. Lower values preserve local structure,
/// which is desirable for document similarity visualization.
/// Must satisfy: n > 3 * perplexity, so this works for n >= 46.
const TSNE_MAX_PERPLEXITY: f64 = 15.0;

/// t-SNE approximation threshold for the Barnes-Hut algorithm.
/// 0.5 provides a good balance between visualization quality and speed.
const TSNE_APPROX_THRESHOLD: f64 = 0.5;

/// Assigns 2D (x, y) coordinates to documents using t-SNE projection.
///
/// Falls back to PCA when there are too few documents for t-SNE, and to
/// a grid layout when no embeddings are present.
pub fn assign_2d_positions(documents: &mut [Document]) {
    if documents.is_empty() {
        return;
    }

    let dim = documents
        .iter()
        .find(|d| !d.embedding.is_empty())
        .map_or(0, |d| d.embedding.len());

    if dim == 0 {
        // No embeddings — use a simple grid layout
        let cols = (documents.len() as f64).sqrt().ceil() as usize;
        for (i, doc) in documents.iter_mut().enumerate() {
            doc.x = (i % cols) as f64;
            doc.y = (i / cols) as f64;
        }
        normalize_positions(documents);
        return;
    }

    let n = documents.len();

    // Build flat f64 embedding matrix (n × dim)
    let flat: Vec<f64> = documents
        .iter()
        .flat_map(|d| {
            if d.embedding.len() == dim {
                d.embedding
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>()
            } else {
                vec![0.0_f64; dim]
            }
        })
        .collect();

    if n >= TSNE_MIN_SAMPLES {
        match run_tsne(&flat, n, dim) {
            Ok(coords) => {
                for (i, doc) in documents.iter_mut().enumerate() {
                    doc.x = coords[i * 2];
                    doc.y = coords[i * 2 + 1];
                }
                normalize_positions(documents);
                return;
            }
            Err(err) => {
                tracing::warn!(error = %err, "t-SNE failed; falling back to PCA");
            }
        }
    } else {
        tracing::debug!(n, "too few documents for t-SNE; using PCA");
    }

    // PCA fallback
    pca_positions(documents, &flat, n, dim);
}

/// Stack size for the t-SNE worker thread (32 MB).
///
/// linfa-tsne allocates large intermediate arrays on the stack during
/// the Barnes-Hut approximation. Debug builds especially need extra headroom.
/// 32 MB handles large document collections (1000+ docs) in debug mode.
/// Release builds use significantly less stack due to optimizations.
const TSNE_STACK_SIZE: usize = 32 * 1024 * 1024;

/// Runs t-SNE and returns a flat `[x0, y0, x1, y1, ...]` coordinate array.
///
/// Spawns a dedicated thread with a larger stack to avoid overflow.
fn run_tsne(flat: &[f64], n: usize, dim: usize) -> eyre::Result<Vec<f64>> {
    let flat = flat.to_vec();
    let handle = std::thread::Builder::new()
        .name("tsne-worker".into())
        .stack_size(TSNE_STACK_SIZE)
        .spawn(move || run_tsne_inner(&flat, n, dim))
        .map_err(|e| eyre::eyre!("failed to spawn t-SNE thread: {e}"))?;

    handle
        .join()
        .map_err(|_| eyre::eyre!("t-SNE thread panicked"))?
}

fn run_tsne_inner(flat: &[f64], n: usize, dim: usize) -> eyre::Result<Vec<f64>> {
    let data = Array2::from_shape_vec((n, dim), flat.to_vec())
        .map_err(|e| eyre::eyre!("ndarray shape error: {e}"))?;

    // Perplexity must satisfy: n > 3 * perplexity.
    // Clamp to stay safely within bounds (floor(n/3) - 1, minimum 1).
    let safe_perplexity = TSNE_MAX_PERPLEXITY.min(((n / 3).saturating_sub(1).max(1)) as f64);

    tracing::debug!(n, perplexity = safe_perplexity, "running t-SNE");

    // TSneParams::embedding_size defaults to SmallRng seeded at 42 — deterministic output.
    // F = f64 is inferred from the Array2<f64> data; R = SmallRng from the impl.
    let result = TSneParams::embedding_size(2)
        .perplexity(safe_perplexity)
        .approx_threshold(TSNE_APPROX_THRESHOLD)
        .transform(data)
        .map_err(|e| eyre::eyre!("t-SNE error: {e:?}"))?;

    let coords: Vec<f64> = result.iter().copied().collect();
    Ok(coords)
}

/// PCA projection fallback (power iteration for 2 principal components).
fn pca_positions(documents: &mut [Document], flat: &[f64], n: usize, dim: usize) {
    let embeddings_f64: Vec<Vec<f64>> = flat.chunks(dim).map(<[f64]>::to_vec).collect();

    // Compute mean
    let n_f = n as f64;
    let mut mean = vec![0.0_f64; dim];
    for emb in &embeddings_f64 {
        for (j, val) in emb.iter().enumerate() {
            mean[j] += val / n_f;
        }
    }

    // Center data
    let centered: Vec<Vec<f64>> = embeddings_f64
        .iter()
        .map(|emb| emb.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();

    let pc1 = power_iteration(&centered, dim);
    let projections1: Vec<f64> = centered.iter().map(|row| dot(row, &pc1)).collect();

    let residual: Vec<Vec<f64>> = centered
        .iter()
        .zip(&projections1)
        .map(|(row, &p)| row.iter().zip(&pc1).map(|(a, b)| a - p * b).collect())
        .collect();

    let pc2 = power_iteration(&residual, dim);
    let projections2: Vec<f64> = residual.iter().map(|row| dot(row, &pc2)).collect();

    for (i, doc) in documents.iter_mut().enumerate() {
        doc.x = projections1[i];
        doc.y = projections2[i];
    }

    normalize_positions(documents);
}

/// Projects all documents into 2D space without modifying cluster IDs.
pub fn project_all(documents: &mut [Document]) {
    assign_2d_positions(documents);
}

/// Normalizes positions to [0, 1] range.
fn normalize_positions(documents: &mut [Document]) {
    if documents.is_empty() {
        return;
    }

    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for doc in documents.iter() {
        min_x = min_x.min(doc.x);
        max_x = max_x.max(doc.x);
        min_y = min_y.min(doc.y);
        max_y = max_y.max(doc.y);
    }

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;

    for doc in documents.iter_mut() {
        doc.x = if range_x > 0.0 {
            (doc.x - min_x) / range_x
        } else {
            0.5
        };
        doc.y = if range_y > 0.0 {
            (doc.y - min_y) / range_y
        } else {
            0.5
        };
    }
}

/// Simple power iteration to approximate the dominant eigenvector.
fn power_iteration(data: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut vec = vec![1.0; dim];
    let norm = (dim as f64).sqrt();
    for v in &mut vec {
        *v /= norm;
    }

    for _ in 0..50 {
        let mut new_vec = vec![0.0; dim];

        // Project data onto current vector
        let projections: Vec<f64> = data.iter().map(|row| dot(row, &vec)).collect();

        // Accumulate back
        for (row, &p) in data.iter().zip(&projections) {
            for (j, val) in row.iter().enumerate() {
                new_vec[j] += p * val;
            }
        }

        // Normalize
        let n = dot(&new_vec, &new_vec).sqrt();
        if n > 0.0 {
            for v in &mut new_vec {
                *v /= n;
            }
        }

        vec = new_vec;
    }

    vec
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
