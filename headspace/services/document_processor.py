"""
Document Processor for Headspace System
Handles document chunking, embedding generation, and position calculation
"""

import time
import math
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Callable
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from data_models import Document, Chunk, ChunkConnection


class DocumentProcessor:
    """Processes documents into chunks with embeddings and spatial positioning"""

    CLUSTER_COLOR_PALETTE = [
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

    def __init__(self, db, embedder, tag_engine, llm_chunker, config_manager, monitor):
        self.db = db
        self.embedder = embedder
        self.tag_engine = tag_engine
        self.llm_chunker = llm_chunker
        self.config_manager = config_manager
        self.monitor = monitor

    def _chunk_document(self, content: str, doc_type: str) -> List[Dict]:
        """Chunk content into structured segments using configured strategy."""
        strategy = self.config_manager.config.get("chunking_strategy", {}).get("preferred_chunker", "llm")
        chunks_data: List[Dict] = []
        chunking_start = time.time()

        if strategy == "llm":
            try:
                self.monitor.logger.debug("Attempting LLM chunking")
                chunks_data = self.llm_chunker.chunk(content)
                for chunk in chunks_data:
                    chunk['type'] = chunk.get('type', 'llm')
                chunking_time = (time.time() - chunking_start) * 1000
                self.monitor.record_model_usage("chunker-main", True, chunking_time)
                self.monitor.logger.info(f"✓ LLM chunking successful: {len(chunks_data)} chunks ({chunking_time:.2f}ms)")
            except Exception as e:
                chunking_time = (time.time() - chunking_start) * 1000
                self.monitor.record_model_usage("chunker-main", False, chunking_time, str(e))
                self.monitor.logger.warning(f"✗ LLM chunking failed, using structural fallback: {e}")
                chunks_data = self._chunk_structural(content, doc_type)
        else:
            self.monitor.logger.debug("Using structural chunking strategy")
            chunks_data = self._chunk_structural(content, doc_type)

        return chunks_data

    def _generate_placeholder_positions(self, count: int) -> List[List[float]]:
        """Generate initial placeholder positions for chunks before enrichment."""
        if count <= 0:
            return []
        if count == 1:
            return [[0.0, 0.0, 0.0]]

        radius = max(24.0, 12.0 + count * 1.5)
        positions: List[List[float]] = []
        for i in range(count):
            angle = (2 * math.pi * i) / count
            y = (i % 5 - 2) * 4.5
            positions.append([math.cos(angle) * radius, y, math.sin(angle) * radius])
        return positions

    def _create_sequential_connections(self, saved_chunks: List[Chunk]):
        """Create simple sequential connections between placeholder chunks."""
        for i in range(len(saved_chunks) - 1):
            connection = ChunkConnection(
                from_chunk_id=saved_chunks[i].id,
                to_chunk_id=saved_chunks[i + 1].id,
                connection_type="sequential",
                strength=1.0,
            )
            self.db.save_connection(connection)

    def create_document_placeholders(self, title: str, content: str, doc_type: str = "text") -> tuple[str, List[Chunk]]:
        """Create a document and chunk placeholders prior to enrichment."""
        doc_id = hashlib.md5(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            doc_type=doc_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "word_count": len(content.split()),
                "status": "processing",
            },
        )
        self.db.save_document(document)
        self.monitor.logger.info(f"📄 Preparing document placeholders: {title} (ID: {doc_id})")

        chunks_data = self._chunk_document(content, doc_type)
        if not chunks_data:
            self.monitor.logger.warning(f"No chunks extracted from document {doc_id}")
            return doc_id, []

        placeholder_positions = self._generate_placeholder_positions(len(chunks_data))
        saved_chunks: List[Chunk] = []

        for i, chunk_data in enumerate(chunks_data):
            placeholder = Chunk(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                chunk_index=i,
                content=chunk_data['text'],
                chunk_type=chunk_data.get('type', 'paragraph'),
                embedding=[],
                position_3d=placeholder_positions[i],
                umap_coordinates=placeholder_positions[i],
                color=self.CLUSTER_COLOR_PALETTE[i % len(self.CLUSTER_COLOR_PALETTE)],
                metadata={**chunk_data.get('metadata', {}), "status": "pending"},
                tags=[],
                tag_confidence={},
                reasoning=chunk_data.get('reasoning', ''),
                embedding_model="",
                nearest_chunk_ids=[],
            )
            self.db.save_chunk(placeholder)
            saved_chunks.append(placeholder)

        document.metadata["chunk_count"] = len(chunks_data)
        document.updated_at = datetime.now()
        self.db.save_document(document)

        self._create_sequential_connections(saved_chunks)
        self.monitor.logger.info(f"✅ Created {len(saved_chunks)} placeholders for document {doc_id}")
        return doc_id, saved_chunks

    def enrich_document(
        self,
        doc_id: str,
        chunk_callback: Optional[Callable[[Chunk, int, int, str], None]] = None,
    ) -> None:
        """Enrich an existing document by generating embeddings, positions, and metadata."""
        document = self.db.get_document(doc_id)
        if not document:
            self.monitor.logger.warning(f"Document {doc_id} not found for enrichment")
            return

        chunks = self.db.get_chunks_by_document(doc_id)
        if not chunks:
            self.monitor.logger.warning(f"No chunks found for document {doc_id} during enrichment")
            return

        chunk_texts = [chunk.content for chunk in chunks]
        embedding_start = time.time()
        try:
            self.monitor.logger.debug(f"Generating embeddings for {len(chunk_texts)} chunks (async pipeline)")
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            embedding_time = (time.time() - embedding_start) * 1000
            self.monitor.record_model_usage("embedder-main", True, embedding_time)
            self.monitor.logger.info(f"✓ Generated {len(embeddings)} embeddings ({embedding_time:.2f}ms)")
        except Exception as e:
            embedding_time = (time.time() - embedding_start) * 1000
            self.monitor.record_model_usage("embedder-main", False, embedding_time, str(e))
            self.monitor.logger.error(f"✗ Embedding generation failed: {e}")
            raise

        embeddings_array = np.array(embeddings)
        positions_3d = self._calculate_3d_positions(embeddings_array)
        saved_chunks: List[Chunk] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Tag generation
            tagging_start = time.time()
            try:
                tag_results = self.tag_engine.generate_tags(chunk.content)
                tagging_time = (time.time() - tagging_start) * 1000
                self.monitor.record_model_usage("tagger-main", True, tagging_time)
            except Exception as e:
                tagging_time = (time.time() - tagging_start) * 1000
                self.monitor.record_model_usage("tagger-main", False, tagging_time, str(e))
                self.monitor.logger.debug(f"Tag generation failed for chunk {i}: {e}")
                tag_results = {}

            embedding_vector = embeddings_array[i].tolist()
            position_vector = positions_3d[i].tolist()

            chunk.embedding = embedding_vector
            chunk.position_3d = position_vector
            chunk.umap_coordinates = position_vector
            chunk.color = self._get_chunk_color(embedding_vector)
            chunk.tags = list(tag_results.keys())
            chunk.tag_confidence = tag_results
            chunk.embedding_model = self.embedder.model_name
            chunk.metadata = {
                **chunk.metadata,
                "status": "enriching",
            }
            chunk.timestamp_modified = datetime.now()
            self.db.save_chunk(chunk)
            saved_chunks.append(chunk)

            if chunk_callback:
                chunk_callback(chunk, i, total_chunks, "embedding")

        self._create_connections(saved_chunks, embeddings_array)
        self._apply_semantic_layout(saved_chunks, embeddings_array, fallback_positions=positions_3d)

        for i, chunk in enumerate(saved_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "status": "enriched",
            }
            self.db.save_chunk(chunk)
            if chunk_callback:
                chunk_callback(chunk, i, total_chunks, "layout")

        # Update document metadata
        document.metadata["status"] = "enriched"
        document.metadata["enriched_at"] = datetime.now().isoformat()
        document.metadata["chunk_count"] = total_chunks
        document.updated_at = datetime.now()
        self.db.save_document(document)

    def process_document_instant(self, title: str, content: str, doc_type: str = "text") -> str:
        """DEPRECATED: Use process_document instead. Kept for compatibility only."""
        self.monitor.logger.warning(f"⚠️  process_document_instant called - using full sync processing instead")
        return self.process_document(title, content, doc_type)

    def process_document(self, title: str, content: str, doc_type: str = "text") -> str:
        """Synchronous processing pipeline used for seeding and tests."""
        doc_id, _ = self.create_document_placeholders(title, content, doc_type)
        self.enrich_document(doc_id)
        return doc_id

    def _chunk_structural(self, content: str, doc_type: str) -> List[Dict]:
        """Wrapper for old structural chunking methods."""
        if doc_type == "code":
            return self._chunk_code(content)
        return self._chunk_text(content)

    def _create_connections(self, saved_chunks: List[Chunk], embeddings: np.ndarray):
        """Create sequential and semantic connections between chunks."""
        # Sequential connections
        for i in range(len(saved_chunks) - 1):
            connection = ChunkConnection(
                from_chunk_id=saved_chunks[i].id, to_chunk_id=saved_chunks[i + 1].id,
                connection_type="sequential", strength=1.0
            )
            self.db.save_connection(connection)

        # Semantic connections
        similarities = self._calculate_similarities(embeddings)
        for i in range(len(saved_chunks)):
            for j in range(i + 1, len(saved_chunks)):
                if similarities[i][j] > 0.8 and abs(i - j) > 1:
                    connection = ChunkConnection(
                        from_chunk_id=saved_chunks[i].id, to_chunk_id=saved_chunks[j].id,
                        connection_type="semantic", strength=float(similarities[i][j])
                    )
                    self.db.save_connection(connection)

    def _chunk_text(self, content: str) -> List[Dict]:
        """Chunk text content by paragraphs and simple rules."""
        chunks = []
        for i, para in enumerate(content.split('\n\n')):
            if not para.strip():
                continue
            chunk_type = 'paragraph'
            if para.startswith('#'):
                level = len(para.split()[0])
                chunk_type = f'heading_{level}'
                para = para.lstrip('#').strip()
            elif para.startswith('```'):
                chunk_type = 'code'
                para = para.strip('`').strip()
            elif para.startswith(('- ', '* ', '1. ')):
                chunk_type = 'list'
            chunks.append({'text': para, 'type': chunk_type, 'metadata': {'index': i}})
        return chunks

    def _chunk_code(self, content: str) -> List[Dict]:
        """Chunk code content by functions or line count."""
        chunks, current_chunk, current_type = [], [], 'code'
        for line in content.split('\n'):
            if any(keyword in line for keyword in ['def ', 'function ', 'fn ', 'func ']):
                if current_chunk:
                    chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
                    current_chunk = []
                current_type = 'function'
            current_chunk.append(line)
            if len(current_chunk) >= 20:
                chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
                current_chunk, current_type = [], 'code'
        if current_chunk:
            chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
        return chunks

    def _get_chunk_color(self, embedding: List[float]) -> str:
        """Generate color from embedding using first 3 principal components"""
        if not embedding or len(embedding) < 3:
            return '#748ffc'  # Default purple

        # Use first 3 dimensions as RGB basis
        # Normalize to 0-1 range
        r = (embedding[0] + 1) / 2 if len(embedding) > 0 else 0.5
        g = (embedding[1] + 1) / 2 if len(embedding) > 1 else 0.5
        b = (embedding[2] + 1) / 2 if len(embedding) > 2 else 0.5

        # Apply subtle transformation to avoid muddy colors
        # Boost the dominant channel
        max_val = max(r, g, b)
        if max_val > 0:
            r = r / max_val * 0.8 + 0.2
            g = g / max_val * 0.8 + 0.2
            b = b / max_val * 0.8 + 0.2

        # Convert to hex
        r_hex = format(int(r * 255), '02x')
        g_hex = format(int(g * 255), '02x')
        b_hex = format(int(b * 255), '02x')

        return f'#{r_hex}{g_hex}{b_hex}'

    def _calculate_3d_positions(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate 3D positions from embeddings using PCA dimensionality reduction."""
        n = len(embeddings)

        if n <= 3:
            # For very few chunks, use simple spacing
            positions = np.zeros((n, 3))
            for i in range(n):
                angle = i * 2 * np.pi / n
                positions[i] = [20 * np.cos(angle), 20 * np.sin(angle), 0]
            return positions

        # Use PCA to reduce embeddings to 3D space
        from sklearn.decomposition import PCA

        # Reduce to 3 dimensions, preserving semantic relationships
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(embeddings)

        # Scale to reasonable viewing range (-50 to 50)
        min_pos = positions_3d.min()
        max_pos = positions_3d.max()
        if max_pos > min_pos:
            positions_3d = (positions_3d - min_pos) / (max_pos - min_pos) * 100 - 50

        return positions_3d

    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between embeddings."""
        n = len(embeddings)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    dot = np.dot(embeddings[i], embeddings[j])
                    norm1, norm2 = np.linalg.norm(embeddings[i]), np.linalg.norm(embeddings[j])
                    if norm1 > 0 and norm2 > 0:
                        sim = dot / (norm1 * norm2)
                        similarities[i][j] = similarities[j][i] = sim
        return similarities

    def _apply_semantic_layout(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        fallback_positions: Optional[np.ndarray] = None,
    ):
        """Apply UMAP + HDBSCAN layout to the given chunks for small-scale visualization."""
        if not chunks:
            return

        num_chunks = len(chunks)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != num_chunks:
            self.monitor.logger.debug("Embedding array shape mismatch; skipping semantic layout.")
            return

        if num_chunks < 2:
            for chunk in chunks:
                coords = chunk.position_3d or chunk.umap_coordinates or [0.0, 0.0, 0.0]
                chunk.umap_coordinates = coords
                chunk.cluster_id = None
                chunk.cluster_confidence = None
                chunk.cluster_label = None
                chunk.nearest_chunk_ids = []
                chunk.metadata["cluster_label"] = None
                self.db.save_chunk(chunk)
            return

        try:
            import umap
            import hdbscan
        except ImportError:
            self.monitor.logger.warning("UMAP/HDBSCAN not installed; skipping semantic layout.")
            return

        n_neighbors = max(2, min(5, num_chunks - 1))
        min_dist = 0.05 if num_chunks <= 10 else 0.1

        try:
            reducer = umap.UMAP(
                n_components=3,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
            )
            umap_coords = reducer.fit_transform(embeddings_array)
        except Exception as exc:
            self.monitor.logger.warning(f"UMAP computation failed ({exc}); retaining PCA layout.")
            if fallback_positions is not None:
                umap_coords = np.array(fallback_positions, dtype=np.float32)
            else:
                return

        if umap_coords.shape[1] != 3:
            self.monitor.logger.debug("UMAP did not return 3D coordinates; using fallback positions.")
            if fallback_positions is not None:
                umap_coords = np.array(fallback_positions, dtype=np.float32)
            else:
                return

        ranges = np.ptp(umap_coords, axis=0)
        max_range = float(np.max(ranges)) if ranges.size else 0.0
        if max_range <= 1e-6:
            if fallback_positions is not None:
                self.monitor.logger.debug("UMAP coordinates collapsed; reverting to fallback positions.")
                umap_coords = np.array(fallback_positions, dtype=np.float32)
            else:
                self.monitor.logger.debug("UMAP coordinates collapsed and no fallback provided; keeping origin.")
                umap_coords = np.zeros_like(umap_coords)
        else:
            centered = umap_coords - umap_coords.mean(axis=0)
            scale = 50.0 / max_range
            umap_coords = centered * scale

        min_cluster_size = max(2, min(5, num_chunks))
        min_samples = max(1, min(3, num_chunks // 2 or 1))

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            clusterer.fit(umap_coords)
            labels = clusterer.labels_
            probabilities = clusterer.probabilities_
        except Exception as exc:
            self.monitor.logger.warning(f"HDBSCAN failed ({exc}); marking all chunks as noise.")
            labels = np.full(num_chunks, -1, dtype=int)
            probabilities = np.zeros(num_chunks, dtype=float)

        try:
            neighbor_count = min(num_chunks, 5)
            nn_model = NearestNeighbors(metric="cosine", algorithm="auto", n_neighbors=neighbor_count)
            nn_model.fit(embeddings_array)
            _, neighbor_indices = nn_model.kneighbors(embeddings_array, return_distance=True)
        except Exception as exc:
            self.monitor.logger.debug(f"Nearest neighbor computation failed ({exc}); using empty neighbor lists.")
            neighbor_indices = np.array([[]] * num_chunks)

        try:
            texts = [chunk.content for chunk in chunks]
            vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            cluster_labels = {}

            for cluster_id in np.unique(labels):
                if cluster_id < 0:
                    continue
                indices = np.where(labels == cluster_id)[0]
                if indices.size == 0:
                    continue
                cluster_tfidf = tfidf_matrix[indices].mean(axis=0)
                scores = np.asarray(cluster_tfidf).ravel()
                top_indices = scores.argsort()[::-1][:3]
                terms = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
                if terms:
                    cluster_labels[cluster_id] = ", ".join(terms)
        except Exception as exc:
            self.monitor.logger.debug(f"TF-IDF cluster labeling failed ({exc}); using fallback labels.")
            cluster_labels = {}

        for idx, chunk in enumerate(chunks):
            coords = umap_coords[idx].tolist()
            chunk.umap_coordinates = coords
            chunk.position_3d = coords

            label_val = int(labels[idx]) if idx < len(labels) else -1
            if label_val >= 0:
                chunk.cluster_id = label_val
                chunk.cluster_confidence = float(probabilities[idx]) if idx < len(probabilities) else None
                chunk.cluster_label = cluster_labels.get(label_val, f"Cluster {label_val}")
                chunk.color = self.CLUSTER_COLOR_PALETTE[label_val % len(self.CLUSTER_COLOR_PALETTE)]
            else:
                chunk.cluster_id = None
                chunk.cluster_confidence = None
                chunk.cluster_label = None

            neighbors = []
            if neighbor_indices.size and idx < len(neighbor_indices):
                for neighbor_idx in neighbor_indices[idx]:
                    neighbor_idx = int(neighbor_idx)
                    if neighbor_idx == idx or neighbor_idx < 0 or neighbor_idx >= num_chunks:
                        continue
                    neighbor_id = chunks[neighbor_idx].id
                    if neighbor_id not in neighbors:
                        neighbors.append(neighbor_id)
            chunk.nearest_chunk_ids = neighbors[:4]

            chunk.metadata["cluster_label"] = chunk.cluster_label

            self.db.save_chunk(chunk)

    def _get_color_from_embedding(self, embedding: np.ndarray) -> str:
        """Generate a hex color from an embedding vector."""
        if len(embedding) >= 3:
            norm_emb = embedding[:3] / np.linalg.norm(embedding[:3]) if np.linalg.norm(embedding[:3]) > 0 else \
            embedding[:3]
            r, g, b = [int((val * 0.5 + 0.5) * 255) for val in norm_emb]
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#667eea"

    def _get_shape_from_tags(self, tags: List[str]) -> str:
        """Determines the 3D shape for a chunk based on its tags."""
        if "code" in tags:
            return "cube"
        if "philosophy" in tags:
            return "icosahedron"
        if "visualization" in tags:
            return "torus"
        return "sphere"