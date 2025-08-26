import time
from typing import List
from globule.core.interfaces import BaseEmbeddingAdapter
from globule.core.models import EmbeddingResult
from globule.core.errors import EmbeddingError
from .ollama_provider import OllamaEmbeddingProvider
from aiohttp import ClientError
import numpy as np


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Ollama embedding adapter implementing BaseEmbeddingAdapter interface."""
    
    def __init__(self, provider: OllamaEmbeddingProvider):
        self._provider = provider
    
    async def embed_single(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text using Ollama provider."""
        start_time = time.time()
        try:
            embedding_vector = await self._provider.embed(text)
            # Handle both numpy arrays and lists
            if hasattr(embedding_vector, 'tolist'):
                embedding_vector = embedding_vector.tolist()
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embedding=embedding_vector,
                dimensions=len(embedding_vector),
                model_name=self.get_model_name(),
                processing_time_ms=processing_time,
                metadata={
                    "provider": "ollama",
                    "base_url": self._provider.base_url,
                    "timeout": self._provider.timeout
                }
            )
        except (RuntimeError, ClientError) as e:
            raise EmbeddingError(f"Ollama embedding provider failed: {e}") from e
    
    async def batch_embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently using the provider's batch support."""
        start_time = time.time()
        try:
            embedding_vectors = await self._provider.embed_batch(texts)
            
            processing_time = (time.time() - start_time) * 1000
            
            results = []
            for vector in embedding_vectors:
                # Handle both numpy arrays and lists
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                
                results.append(EmbeddingResult(
                    embedding=vector,
                    dimensions=len(vector),
                    model_name=self.get_model_name(),
                    processing_time_ms=processing_time / len(embedding_vectors), # Distribute total time
                    metadata={
                        "provider": "ollama",
                        "batch_size": len(texts)
                    }
                ))
            return results
        except (RuntimeError, ClientError) as e:
            raise EmbeddingError(f"Ollama batch embedding failed: {e}") from e
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for this model by asking the provider."""
        # Defer to the provider, which caches the dimension after the first call.
        # This is more robust than a hardcoded value.
        return self._provider.get_dimension()
    
    def get_model_name(self) -> str:
        """Return the name of the embedding model being used."""
        return self._provider.model