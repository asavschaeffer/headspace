#!/usr/bin/env python3
"""
Keyword search engine for chunk retrieval.
Supports full-text search, keyword extraction, and TF-IDF based ranking.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import math
from tag_keywords import KEYWORDS


@dataclass
class KeywordMatch:
    """Result of a keyword search."""

    chunk_id: str
    score: float  # Relevance score (0-1)
    matched_keywords: List[str]
    match_positions: List[int]  # Character positions of matches


class KeywordExtractor:
    """Extracts keywords and key phrases from text."""

    def __init__(self, min_word_length: int = 3):
        """
        Initialize keyword extractor.

        Args:
            min_word_length: Minimum word length to consider
        """
        self.min_word_length = min_word_length
        self.stop_words = self._build_stop_words()

    def _build_stop_words(self) -> Set[str]:
        """Build common English stop words."""
        common_stops = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "is",
            "was",
            "are",
            "be",
            "been",
            "that",
            "this",
            "it",
            "by",
            "from",
            "as",
            "if",
            "can",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
        }
        return common_stops

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from text using TF-IDF heuristic.

        Args:
            text: Input text
            top_k: Number of top keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        # Normalize text
        text = text.lower()

        # Extract words
        words = re.findall(r"\b[a-z]+\b", text)
        words = [w for w in words if len(w) >= self.min_word_length and w not in self.stop_words]

        if not words:
            return []

        # Calculate term frequency
        word_freq = Counter(words)
        total_words = len(words)

        # Score based on frequency and word importance
        scored_words = []
        for word, count in word_freq.items():
            # TF score
            tf = count / total_words if total_words > 0 else 0

            # Boost score if word appears in predefined keywords
            boost = 1.0
            if hasattr(self, "predefined_keywords"):
                if any(word in kw.lower() for kw in self.predefined_keywords):
                    boost = 1.5

            score = tf * boost
            scored_words.append((word, score))

        # Sort by score and return top_k
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return scored_words[:top_k]

    def extract_phrases(self, text: str, phrase_length: int = 2) -> List[str]:
        """
        Extract key phrases (consecutive words).

        Args:
            text: Input text
            phrase_length: Length of phrases (default: 2-word phrases)

        Returns:
            List of key phrases
        """
        words = re.findall(r"\b[a-z]+\b", text.lower())
        words = [w for w in words if len(w) >= self.min_word_length and w not in self.stop_words]

        phrases = []
        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i : i + phrase_length])
            phrases.append(phrase)

        # Return most frequent phrases
        phrase_freq = Counter(phrases)
        return [p for p, _ in phrase_freq.most_common(10)]


class KeywordSearchEngine:
    """
    Full-text keyword search engine for chunks.
    Supports exact match, fuzzy match, and ranking by relevance.
    """

    def __init__(self):
        """Initialize the search engine."""
        self.extractor = KeywordExtractor()
        self.chunk_index: Dict[str, Dict[str, any]] = {}  # chunk_id -> {keywords, content, position}

    def index_chunk(self, chunk_id: str, content: str):
        """
        Index a chunk for keyword search.

        Args:
            chunk_id: Unique chunk identifier
            content: Chunk content/text
        """
        # Extract keywords from content
        keywords = self.extractor.extract_keywords(content, top_k=15)
        phrases = self.extractor.extract_phrases(content)

        self.chunk_index[chunk_id] = {
            "content": content,
            "keywords": [kw for kw, _ in keywords],
            "keyword_scores": {kw: score for kw, score in keywords},
            "phrases": phrases,
            "word_count": len(content.split()),
            "char_count": len(content),
        }

    def search(
        self, query: str, top_k: int = 10, min_score: float = 0.1
    ) -> List[KeywordMatch]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query (keywords/phrases)
            top_k: Number of results to return
            min_score: Minimum relevance score threshold

        Returns:
            List of KeywordMatch results sorted by relevance
        """
        # Normalize query
        query = query.lower().strip()

        if not query or not self.chunk_index:
            return []

        # Extract query keywords
        query_keywords = set(re.findall(r"\b[a-z]+\b", query))
        query_keywords = {w for w in query_keywords if len(w) >= 3}

        results = []

        # Score each chunk
        for chunk_id, chunk_data in self.chunk_index.items():
            score, matched_kws, positions = self._score_chunk(query_keywords, chunk_data)

            if score >= min_score:
                results.append(
                    KeywordMatch(
                        chunk_id=chunk_id,
                        score=score,
                        matched_keywords=matched_kws,
                        match_positions=positions,
                    )
                )

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _score_chunk(self, query_keywords: Set[str], chunk_data: Dict) -> Tuple[float, List[str], List[int]]:
        """
        Score a chunk against query keywords.

        Returns:
            (score, matched_keywords, match_positions)
        """
        chunk_keywords = set(chunk_data["keywords"])
        content = chunk_data["content"]

        # Find matched keywords
        matched = query_keywords & chunk_keywords
        if not matched:
            return 0.0, [], []

        # Calculate score based on keyword matches
        num_matches = len(matched)
        unique_query_terms = len(query_keywords)

        # Coverage score: how many query terms matched
        coverage = num_matches / unique_query_terms if unique_query_terms > 0 else 0

        # Frequency score: how important are the matched keywords in the chunk
        frequency = sum(chunk_data["keyword_scores"].get(kw, 0) for kw in matched) / len(matched)

        # Combined score
        score = (coverage * 0.6 + frequency * 0.4)

        # Find positions of matches in content
        positions = []
        content_lower = content.lower()
        for kw in matched:
            start = 0
            while True:
                pos = content_lower.find(kw, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

        return min(score, 1.0), list(matched), positions

    def hybrid_search(
        self,
        query: str,
        vector_results: List[Tuple[str, float]],
        top_k: int = 10,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> List[Tuple[str, float]]:
        """
        Combine vector search and keyword search results.

        Args:
            query: Search query
            vector_results: List of (chunk_id, score) from vector search
            top_k: Number of results to return
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)

        Returns:
            List of (chunk_id, combined_score) sorted by combined score
        """
        # Get keyword search results
        keyword_matches = self.search(query, top_k=len(self.chunk_index))
        keyword_results = {m.chunk_id: m.score for m in keyword_matches}

        # Combine scores
        combined_scores = {}

        # Add vector search scores
        for chunk_id, v_score in vector_results:
            combined_scores[chunk_id] = v_score * vector_weight

        # Add keyword search scores
        for chunk_id, k_score in keyword_results.items():
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += k_score * keyword_weight
            else:
                combined_scores[chunk_id] = k_score * keyword_weight

        # Sort and return top results
        results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def batch_index(self, chunks: List[Dict[str, str]]):
        """
        Index multiple chunks at once.

        Args:
            chunks: List of dicts with 'id' and 'content' keys
        """
        for chunk in chunks:
            if "id" in chunk and "content" in chunk:
                self.index_chunk(chunk["id"], chunk["content"])

    def clear_index(self):
        """Clear the search index."""
        self.chunk_index.clear()
