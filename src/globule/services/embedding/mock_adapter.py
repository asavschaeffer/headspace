"""
Mock embedding adapter for testing and development.

Implements the BaseEmbeddingAdapter interface with deterministic mock behavior.
"""
import time
import hashlib
from typing import List
from globule.core.interfaces import BaseEmbeddingAdapter
from globule.core.models import EmbeddingResult
from globule.core.errors import EmbeddingError


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """Mock embedding adapter for testing and Phase 1 development."""
    
    def __init__(self, dimension: int = 1024, model_name: str = "mock-embedder-v1.0"):
        self.dimension = dimension
        self.model_name = model_name
    
    async def embed_single(self, text: str) -> EmbeddingResult:
        """Generate mock embedding for a single text."""
        start_time = time.time()
        
        if not text:
            raise EmbeddingError("Input text cannot be empty")
        
        # Generate consistent embeddings based on text hash for reproducibility
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        # Simple deterministic generation
        embedding = []
        for i in range(self.dimension):
            val = ((seed + i) % 10000) / 10000.0 - 0.5
            embedding.append(val)
        
        # Normalize to unit vector
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        processing_time = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embedding=embedding,
            dimensions=len(embedding),
            model_name=self.model_name,
            processing_time_ms=processing_time,
            metadata={
                "provider": "mock",
                "text_length": len(text),
                "seed": seed,
                "deterministic": True
            }
        )
    
    async def batch_embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate mock embeddings for multiple texts."""
        results = []
        for text in texts:
            result = await self.embed_single(text)
            results.append(result)
        return results
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Return the mock model name."""
        return self.model_name