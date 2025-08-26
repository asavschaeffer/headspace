"""
Unit tests for core logic that doesn't require external dependencies.

These tests verify the core functionality without requiring:
- vec0 SQLite extension
- Ollama service
- External AI services
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from globule.core.models import GlobuleV1, ProcessedGlobuleV1, FileDecisionV1
from globule.orchestration.engine import GlobuleOrchestrator


class TestCoreModels:
    """Test core data models."""
    
    def test_processed_globule_creation(self):
        """Test ProcessedGlobuleV1 can be created with minimal data."""
        raw_globule = GlobuleV1(raw_text="Test thought", source="test")
        processed = ProcessedGlobuleV1(
            globule_id=raw_globule.globule_id,
            original_globule=raw_globule,
            embedding=[0.1, 0.2],
            processing_time_ms=50.0,
            parsed_data={"title": "Test"}
        )
        
        assert processed.original_globule.raw_text == "Test thought"
        assert processed.parsed_data["title"] == "Test"
        assert processed.globule_id == raw_globule.globule_id
    
    def test_file_decision_creation(self):
        """Test FileDecision creation."""
        decision = FileDecisionV1(
            semantic_path="notes/personal",
            filename="test.md",
            confidence=0.9
        )
        
        assert decision.semantic_path == "notes/personal"
        assert decision.filename == "test.md"
        assert decision.confidence == 0.9


class TestFilePath:
    """Test file path generation logic."""
    
    def test_file_decision_path_construction(self):
        """Test that file paths are constructed correctly."""
        decision = FileDecisionV1(
            semantic_path="projects/ai",
            filename="neural-networks.md",
            confidence=0.85
        )
        
        # Reconstruct path for verification
        full_path = Path(decision.semantic_path) / decision.filename
        # Use forward slashes for cross-platform compatibility
        assert str(full_path).replace("\\", "/") == "projects/ai/neural-networks.md"
        assert full_path.suffix == ".md"
        assert full_path.stem == "neural-networks"



class TestVectorOperations:
    """Test vector operations without requiring database."""
    
    def test_vector_normalization(self):
        """Test vector normalization logic."""
        # Create a test vector
        vector = np.array([3.0, 4.0, 0.0])
        
        # Normalize it
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 0 else vector
        
        # Check properties
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        assert normalized.shape == vector.shape
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Create test vectors
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        vec_c = np.array([1.0, 0.0, 0.0])
        
        # Calculate similarities
        def cosine_similarity(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        
        # Test cases
        assert np.isclose(cosine_similarity(vec_a, vec_c), 1.0)  # Same direction
        assert np.isclose(cosine_similarity(vec_a, vec_b), 0.0)  # Orthogonal
        assert np.isclose(cosine_similarity(vec_a, -vec_a), -1.0)  # Opposite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])