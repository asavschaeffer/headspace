#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use hdbscan::Hdbscan;

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
                vec![0.0; 1024]
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

/// Assigns 2D (x, y) coordinates to documents using simple PCA on embeddings.
///
/// Uses f32 embeddings internally, upcast to f64 for the linear algebra
/// (PCA needs the precision for eigenvalue convergence).
fn assign_2d_positions(documents: &mut [Document]) {
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

    // Upcast embeddings to f64 for PCA math
    let embeddings_f64: Vec<Vec<f64>> = documents
        .iter()
        .map(|d| {
            if d.embedding.len() == dim {
                d.embedding.iter().map(|&v| f64::from(v)).collect()
            } else {
                vec![0.0; dim]
            }
        })
        .collect();

    // Compute mean
    let n = documents.len() as f64;
    let mut mean = vec![0.0; dim];
    for emb in &embeddings_f64 {
        for (j, val) in emb.iter().enumerate() {
            mean[j] += val / n;
        }
    }

    // Center data
    let centered: Vec<Vec<f64>> = embeddings_f64
        .iter()
        .map(|emb| emb.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();

    // Power iteration for first principal component
    let pc1 = power_iteration(&centered, dim);
    let projections1: Vec<f64> = centered.iter().map(|row| dot(row, &pc1)).collect();

    // Compute residual for PC2
    let residual: Vec<Vec<f64>> = centered
        .iter()
        .zip(&projections1)
        .map(|(row, &p)| row.iter().zip(&pc1).map(|(a, b)| a - p * b).collect())
        .collect();

    let pc2 = power_iteration(&residual, dim);
    let projections2: Vec<f64> = residual.iter().map(|row| dot(row, &pc2)).collect();

    // Assign positions
    for (i, doc) in documents.iter_mut().enumerate() {
        doc.x = projections1[i];
        doc.y = projections2[i];
    }

    normalize_positions(documents);
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
