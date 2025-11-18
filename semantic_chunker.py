#!/usr/bin/env python3
"""
Hybrid semantic chunker using embeddings for boundary detection.
Chunks on paragraph breaks for structure preservation,
then applies semantic splitting within long paragraphs.
No LLM calls - uses embedding similarity analysis.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from embeddings_engine import EmbeddingsEngine
from config_manager import ConfigManager


@dataclass
class ChunkBoundary:
    """Represents a chunk boundary with semantic confidence."""
    position: int  # Character position in text
    similarity_score: float  # Cosine similarity score (0-1)
    reason: str  # Why this boundary was detected


class SemanticChunker:
    """
    Hybrid chunker that preserves paragraph structure while using embeddings
    to detect semantic boundaries within long paragraphs.
    """

    def __init__(
        self,
        embedder: EmbeddingsEngine,
        config_manager: ConfigManager,
        similarity_threshold: float = 0.7,
        max_paragraph_tokens: int = 500,
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedder: EmbeddingsEngine instance for generating embeddings
            config_manager: Configuration manager
            similarity_threshold: Cosine similarity threshold for chunk boundaries (0-1)
            max_paragraph_tokens: Maximum tokens per chunk (rough estimate)
        """
        self.embedder = embedder
        self.config_manager = config_manager
        self.similarity_threshold = similarity_threshold
        self.max_paragraph_tokens = max_paragraph_tokens

    def chunk(self, text: str) -> List[Dict]:
        """
        Chunk text using hybrid approach: paragraphs + semantic boundaries.

        Args:
            text: The text to chunk

        Returns:
            List of chunk dicts with 'text', 'reasoning', and 'semantic_score'
        """
        # Step 1: Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return [{"text": text, "reasoning": "Single paragraph", "type": "semantic"}]

        chunks = []

        # Step 2: Process each paragraph
        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Check if paragraph is long enough to warrant semantic splitting
            token_estimate = len(para.split())

            if token_estimate > self.max_paragraph_tokens:
                # Apply semantic splitting to long paragraphs
                semantic_chunks = self._semantic_split_paragraph(para)
                chunks.extend(semantic_chunks)
            else:
                # Keep short paragraphs intact
                chunks.append(
                    {
                        "text": para,
                        "reasoning": f"Paragraph {para_idx + 1} (structure preserved)",
                        "semantic_score": 1.0,
                        "type": "paragraph",
                    }
                )

        return chunks if chunks else [{"text": text, "reasoning": "Fallback", "type": "semantic"}]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs by double newlines."""
        # Handle various newline formats
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _semantic_split_paragraph(self, paragraph: str) -> List[Dict]:
        """
        Split a paragraph using semantic similarity analysis.
        Detects boundaries where topic changes (low embedding similarity).

        Args:
            paragraph: Text to split

        Returns:
            List of chunk dicts
        """
        # Step 1: Split into sentences
        sentences = self._split_sentences(paragraph)
        if len(sentences) <= 1:
            return [
                {
                    "text": paragraph,
                    "reasoning": "Single sentence",
                    "semantic_score": 1.0,
                    "type": "semantic",
                }
            ]

        # Step 2: Generate embeddings for all sentences
        embeddings = self._get_sentence_embeddings(sentences)

        # Step 3: Find boundaries based on similarity drops
        boundaries = self._detect_boundaries(sentences, embeddings)

        # Step 4: Reconstruct chunks from boundaries
        chunks = self._reconstruct_chunks(sentences, boundaries)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using common delimiters."""
        # Simple sentence splitting on periods, question marks, exclamation marks
        # Preserves sentences with their delimiters
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_sentence_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for sentences.

        Args:
            sentences: List of sentence strings

        Returns:
            List of embedding vectors
        """
        try:
            # Use embedder in batch mode
            embeddings_result = self.embedder.embed(sentences)

            # Convert to numpy arrays if needed
            if isinstance(embeddings_result, list):
                return [np.array(e, dtype=np.float32) for e in embeddings_result]
            elif isinstance(embeddings_result, np.ndarray):
                return [embeddings_result[i] for i in range(len(sentences))]
            else:
                raise ValueError(f"Unexpected embedding format: {type(embeddings_result)}")
        except Exception as e:
            print(f"Warning: Could not generate embeddings: {e}")
            # Fallback: treat all sentences as separate chunks
            return [np.ones(384, dtype=np.float32) for _ in sentences]

    def _detect_boundaries(
        self, sentences: List[str], embeddings: List[np.ndarray]
    ) -> List[ChunkBoundary]:
        """
        Detect semantic boundaries by analyzing cosine similarity between consecutive sentences.

        Args:
            sentences: List of sentences
            embeddings: List of embedding vectors

        Returns:
            List of boundary positions
        """
        boundaries = []

        for i in range(len(embeddings) - 1):
            # Compute cosine similarity
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # Create boundary if similarity drops below threshold
            if similarity < self.similarity_threshold:
                boundaries.append(
                    ChunkBoundary(
                        position=i + 1,  # Boundary after sentence i
                        similarity_score=similarity,
                        reason=f"Topic shift (similarity: {similarity:.2f})",
                    )
                )

        return boundaries

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def _reconstruct_chunks(self, sentences: List[str], boundaries: List[ChunkBoundary]) -> List[Dict]:
        """
        Reconstruct chunks from sentence list and boundaries.

        Args:
            sentences: List of sentences
            boundaries: List of boundary positions

        Returns:
            List of chunk dicts
        """
        if not boundaries:
            # No boundaries detected, return entire paragraph as one chunk
            return [
                {
                    "text": " ".join(sentences),
                    "reasoning": "No semantic boundaries detected",
                    "semantic_score": 1.0,
                    "type": "semantic",
                }
            ]

        chunks = []
        boundary_positions = sorted([b.position for b in boundaries])
        start_idx = 0

        for boundary_pos in boundary_positions:
            # Create chunk from start_idx to boundary_pos
            chunk_sentences = sentences[start_idx:boundary_pos]
            if chunk_sentences:
                chunk_text = " ".join(chunk_sentences)

                # Find the boundary object for this position
                boundary_obj = next((b for b in boundaries if b.position == boundary_pos), None)
                reasoning = boundary_obj.reason if boundary_obj else "Semantic boundary"

                chunks.append(
                    {
                        "text": chunk_text,
                        "reasoning": reasoning,
                        "semantic_score": boundary_obj.similarity_score if boundary_obj else 0.5,
                        "type": "semantic",
                    }
                )

            start_idx = boundary_pos

        # Add final chunk
        if start_idx < len(sentences):
            chunk_text = " ".join(sentences[start_idx:])
            chunks.append(
                {
                    "text": chunk_text,
                    "reasoning": "Final chunk",
                    "semantic_score": 1.0,
                    "type": "semantic",
                }
            )

        return chunks
