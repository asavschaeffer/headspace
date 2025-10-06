import json
import re
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from collections import Counter

class TagEngine:
    """
    Multi-stage intelligent tagging engine.

    Stage 1: Keyword matching (fast, direct)
    Stage 2: Vector clustering (semantic grouping)
    Stage 3: LLM classification (high accuracy, slow)
    """

    def __init__(self, keywords_path: str = "tag_keywords.json",
                 config_path: str = "loom_config.json",
                 default_confidence: float = 0.7):
        """
        Initializes the TagEngine with multi-stage capabilities.

        Args:
            keywords_path: Path to the JSON file containing tag keywords.
            config_path: Path to loom config for LLM settings
            default_confidence: The default confidence score for keyword matches.
        """
        self.keywords_data = self._load_keywords(keywords_path)
        self.config = self._load_config(config_path)
        self.default_confidence = default_confidence
        self.compiled_patterns = self._compile_patterns()

        # LLM settings for Stage 3
        self.ollama_url = self.config.get('ollama', {}).get('url', 'http://localhost:11434/api/generate')
        self.ollama_model = self.config.get('ollama', {}).get('model', 'llama3')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load loom configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _load_keywords(self, keywords_path: str) -> Dict[str, Any]:
        """Loads the keywords from the specified JSON file."""
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Keywords file not found at '{keywords_path}'. Tagging will be disabled.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{keywords_path}'. Tagging will be disabled.")
            return {}

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Compiles regex patterns for each keyword for efficient searching.
        We use word boundaries (\b) to avoid matching substrings inside other words.
        """
        patterns = {}
        if not self.keywords_data:
            return patterns
            
        for tag, data in self.keywords_data.items():
            keyword_patterns = []
            # Combine keywords and context_boost for matching
            all_keywords = data.get("keywords", []) + data.get("context_boost", [])
            for keyword in all_keywords:
                # Escape special regex characters in the keyword
                escaped_keyword = re.escape(keyword)
                # Compile with word boundaries and case-insensitivity
                pattern = re.compile(r'\b' + escaped_keyword + r'\b', re.IGNORECASE)
                keyword_patterns.append(pattern)
            patterns[tag] = keyword_patterns
        return patterns

    def generate_tags(self, text: str, embedding: Optional[np.ndarray] = None,
                      all_embeddings: Optional[np.ndarray] = None,
                      all_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Generates tags for a given text using multi-stage approach.

        Stage 1: Keyword matching (always runs, fast)
        Stage 2: Vector clustering (if embeddings provided)
        Stage 3: LLM classification (if Stage 1 found < 2 tags)

        Args:
            text: The input string to tag.
            embedding: Optional embedding vector for this text (for Stage 2)
            all_embeddings: All embeddings in corpus (for clustering)
            all_texts: All texts in corpus (for clustering)

        Returns:
            A dictionary with found tags as keys and their confidence scores as values.
        """
        # Stage 1: Keyword Matching
        tags = self._stage1_keyword_matching(text)

        # Stage 2: Vector Clustering (if embeddings available)
        if embedding is not None and all_embeddings is not None:
            cluster_tags = self._stage2_vector_clustering(embedding, all_embeddings, all_texts or [])
            # Merge with higher confidence
            for tag, confidence in cluster_tags.items():
                if tag not in tags:
                    tags[tag] = confidence
                else:
                    # Boost confidence if both stages agree
                    tags[tag] = min(tags[tag] + confidence * 0.3, 1.0)

        # Stage 3: LLM Classification (if insufficient tags)
        if len(tags) < 2:
            llm_tags = self._stage3_llm_classification(text, list(tags.keys()))
            for tag, confidence in llm_tags.items():
                tags[tag] = confidence

        return tags

    def _stage1_keyword_matching(self, text: str) -> Dict[str, float]:
        """
        Stage 1: Fast keyword pattern matching

        Returns tags with confidence 0.7 for direct matches
        """
        found_tags = {}
        if not self.compiled_patterns:
            return found_tags

        for tag, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(text):
                    match_count += 1

            if match_count > 0:
                # Higher confidence if multiple keywords match
                confidence = min(0.7 + (match_count - 1) * 0.1, 0.95)
                found_tags[tag] = confidence

        return found_tags

    def _stage2_vector_clustering(self, embedding: np.ndarray,
                                   all_embeddings: np.ndarray,
                                   all_texts: List[str]) -> Dict[str, float]:
        """
        Stage 2: Identify cluster membership and assign cluster-level tags

        Uses DBSCAN to find semantic clusters, then tags based on cluster themes

        Returns tags with confidence 0.8 from clustering
        """
        if len(all_embeddings) < 5:
            return {}  # Need enough samples for clustering

        # Run DBSCAN clustering
        clusterer = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clusterer.fit_predict(all_embeddings)

        # Find which cluster this embedding belongs to
        current_idx = len(all_embeddings) - 1  # Assume current embedding is last
        cluster_id = labels[current_idx]

        if cluster_id == -1:  # Noise point
            return {}

        # Get all texts in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_texts = [all_texts[i] for i in cluster_indices if i < len(all_texts)]

        # Extract most common tags from cluster using Stage 1
        cluster_tags = Counter()
        for text in cluster_texts:
            text_tags = self._stage1_keyword_matching(text)
            for tag in text_tags:
                cluster_tags[tag] += 1

        # Return tags that appear in >50% of cluster
        threshold = len(cluster_texts) * 0.5
        result_tags = {}
        for tag, count in cluster_tags.items():
            if count >= threshold:
                result_tags[tag] = 0.8  # Cluster-level confidence

        return result_tags

    def _stage3_llm_classification(self, text: str, existing_tags: List[str]) -> Dict[str, float]:
        """
        Stage 3: LLM-based classification for ambiguous cases

        Only called when Stage 1 + 2 found < 2 tags

        Returns tags with confidence 0.9 from LLM
        """
        # Get list of all available tags
        available_tags = list(self.keywords_data.keys())

        prompt = f"""Classify this text into the most appropriate tags.

Available tags: {', '.join(available_tags)}

Text to classify:
---
{text[:500]}
---

Select 1-3 most relevant tags from the list above.
If the text truly doesn't fit any existing tags, you may suggest ONE new tag.

Respond ONLY with valid JSON in this format:
{{
  "tags": ["tag1", "tag2"],
  "reasoning": "Brief explanation of why these tags fit"
}}

JSON response:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )

            if response.status_code != 200:
                return {}

            result = response.json()
            response_text = result.get('response', '{}')
            data = json.loads(response_text)

            llm_tags = data.get('tags', [])
            result = {}
            for tag in llm_tags:
                if tag in available_tags or len(existing_tags) == 0:
                    result[tag] = 0.9  # High confidence from LLM

            return result

        except Exception as e:
            print(f"  LLM tagging error: {e}")
            return {}

if __name__ == '__main__':
    # Example usage and testing
    engine = TagEngine()

    test_text_code = "This is a Python function definition: def my_function(): pass"
    test_text_philosophy = "We discussed the metaphysical implications of reality."
    test_text_mixed = "This software project visualizes knowledge as a cosmic universe."

    tags_code = engine.generate_tags(test_text_code)
    tags_philosophy = engine.generate_tags(test_text_philosophy)
    tags_mixed = engine.generate_tags(test_text_mixed)

    print(f"Text: '{test_text_code}'")
    print(f"Found Tags: {tags_code}\n")
    assert "code" in tags_code and "programming" in tags_code

    print(f"Text: '{test_text_philosophy}'")
    print(f"Found Tags: {tags_philosophy}\n")
    assert "philosophy" in tags_philosophy

    print(f"Text: '{test_text_mixed}'")
    print(f"Found Tags: {tags_mixed}\n")
    assert "programming" in tags_mixed and "knowledge" in tags_mixed and "visualization" in tags_mixed
    
    print("TagEngine tests passed.")
