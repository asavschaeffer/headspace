"""
Rebuild global Headspace cosmos embeddings and clustering metadata.

This script computes:
  • 3D UMAP coordinates (cosine metric) for all chunk embeddings
  • HDBSCAN clustering assignments + confidence scores
  • TF-IDF derived cluster labels
  • Cosine nearest-neighbour cache for quick lookup

Results are written back into the SQLite database via DatabaseManager.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "umap-learn is required for rebuild_cosmos. "
        "Install with `pip install umap-learn`."
    ) from exc

try:
    import hdbscan
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "hdbscan is required for rebuild_cosmos. "
        "Install with `pip install hdbscan`."
    ) from exc

from headspace.services.database import DatabaseManager


CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#5254a3",
    "#8ca252",
    "#bd9e39",
    "#ad494a",
    "#a55194",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Headspace cosmos metadata.")
    parser.add_argument("--db-path", default="headspace.db", help="Path to the Headspace SQLite database.")
    parser.add_argument("--n-neighbors", type=int, default=50, help="UMAP: number of neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP: minimum distance.")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP: random state.")
    parser.add_argument("--min-cluster-size", type=int, default=30, help="HDBSCAN: minimum cluster size.")
    parser.add_argument("--min-cluster-samples", type=int, default=None, help="HDBSCAN: minimum samples.")
    parser.add_argument("--nearest-neighbors", type=int, default=12, help="Number of nearest neighbours to cache per chunk.")
    parser.add_argument("--label-terms", type=int, default=3, help="Number of top TF-IDF terms to use for cluster labels.")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum TF-IDF vocabulary size.")
    return parser.parse_args()


def derive_cluster_labels(
    texts: List[str],
    labels: np.ndarray,
    label_terms: int,
    max_features: int,
) -> Dict[int, str]:
    """Compute descriptive labels for clusters using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    label_map: Dict[int, str] = {}

    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue

        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue

        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        scores = np.asarray(cluster_tfidf).ravel()
        if not np.any(scores):
            continue

        top_indices = scores.argsort()[::-1][:label_terms]
        top_terms = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
        if top_terms:
            label_map[cluster_id] = ", ".join(top_terms)

    return label_map


def choose_cluster_color(cluster_id: int | None) -> str | None:
    if cluster_id is None or cluster_id < 0:
        return None
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def rebuild_cosmos(args: argparse.Namespace):
    db = DatabaseManager(db_path=args.db_path)

    print("📦 Loading chunk embeddings…")
    chunks = db.get_all_chunk_embeddings()
    if not chunks:
        print("⚠️  No chunks with embeddings found. Aborting.")
        return

    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        print("⚠️  Embeddings array is empty. Aborting.")
        return

    print(f"🧮 Running UMAP on {len(chunks)} chunks (dim={embeddings.shape[1]})…")
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=min(args.n_neighbors, max(2, len(chunks) - 1)),
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    umap_coords = reducer.fit_transform(embeddings)

    print("🌌 Clustering with HDBSCAN…")
    min_cluster_size = max(2, min(args.min_cluster_size, len(chunks)))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=args.min_cluster_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    try:
        clusterer.fit(umap_coords)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
    except ValueError as exc:
        print(f"⚠️  HDBSCAN failed ({exc}). Marking all chunks as noise.")
        labels = np.full(len(chunks), -1, dtype=int)
        probabilities = np.zeros(len(chunks), dtype=float)

    print("🧾 Deriving cluster labels (TF-IDF)…")
    texts = [
        f"{chunk['document_title']} {chunk['content']}".strip()
        for chunk in chunks
    ]
    cluster_labels = derive_cluster_labels(texts, labels, args.label_terms, args.max_features)

    print("🔎 Building nearest-neighbour cache…")
    nn_count = min(args.nearest_neighbors + 1, len(chunks))
    nn_model = NearestNeighbors(metric="cosine", algorithm="auto", n_neighbors=nn_count)
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings, return_distance=True)

    print("🧽 Clearing old cluster metadata…")
    db.clear_cluster_metadata()

    print("💾 Writing results to database…")
    cluster_sizes: Dict[int, int] = defaultdict(int)
    for label in labels:
        if label >= 0:
            cluster_sizes[label] += 1

    for cluster_id, size in cluster_sizes.items():
        label = cluster_labels.get(cluster_id)
        color = choose_cluster_color(cluster_id)
        db.upsert_cluster_metadata(cluster_id, label, size, color)

    for idx, chunk in enumerate(tqdm(chunks, desc="Updating chunks")):
        chunk_id = chunk["id"]
        label = int(labels[idx])
        probability = float(probabilities[idx]) if probabilities is not None else None
        coords = umap_coords[idx].tolist()
        color = choose_cluster_color(label)
        nearest_indices = [chunks[i]["id"] for i in indices[idx] if chunks[i]["id"] != chunk_id]

        db.update_chunk_cluster_info(
            chunk_id=chunk_id,
            cluster_id=label if label >= 0 else None,
            cluster_confidence=probability if label >= 0 else None,
            cluster_label=cluster_labels.get(label) if label >= 0 else None,
            coordinates=coords,
            nearest_ids=nearest_indices[: args.nearest_neighbors],
            color=color,
        )

    print("✅ Cosmos rebuild complete!")


def main():
    args = parse_args()
    rebuild_cosmos(args)


if __name__ == "__main__":
    main()

