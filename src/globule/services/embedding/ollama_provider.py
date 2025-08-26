"""
Ollama-based embedding provider for Globule.

Implements the EmbeddingProvider interface using Ollama's local API.
"""

import asyncio
import aiohttp
import numpy as np
from typing import List, Optional
import logging

from globule.core.interfaces import IEmbeddingAdapter
from globule.config.settings import get_config

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(IEmbeddingAdapter):
    """Ollama implementation of EmbeddingProvider"""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[int] = None):
        self.config = get_config()
        self.base_url = base_url or self.config.ollama_base_url
        self.model = model or self.config.default_embedding_model
        self.timeout = timeout or self.config.ollama_timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._dimension: Optional[int] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "input": text.strip(),
            "truncate": True
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/embed",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Handle Ollama's response format
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    embedding = np.array(data["embeddings"][0], dtype=np.float32)
                    
                    # Cache dimension on first call
                    if self._dimension is None:
                        self._dimension = len(embedding)
                    
                    return embedding
                elif "embedding" in data:
                    # Alternative response format
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    
                    if self._dimension is None:
                        self._dimension = len(embedding)
                    
                    return embedding
                else:
                    raise RuntimeError(f"Invalid response format from Ollama: {data}")
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts using a single batch request."""
        session = await self._get_session()
        
        # Ollama's /api/embed endpoint supports batching by passing a list of strings.
        payload = {
            "model": self.model,
            "input": [text.strip() for text in texts],
            "truncate": True
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/embed",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    embeddings = [np.array(e, dtype=np.float32) for e in data["embeddings"]]
                    
                    if self._dimension is None and embeddings:
                        self._dimension = len(embeddings[0])
                        
                    return embeddings
                else:
                    raise RuntimeError(f"Invalid response format from Ollama for batch request: {data}")
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        if self._dimension is None:
            # Default dimension for mxbai-embed-large
            return 1024
        return self._dimension
    
    # Interface compliance methods
    async def embed_single(self, text: str):
        """Generate embedding for single text (interface compliance)"""
        from globule.core.models import EmbeddingResult
        import time
        
        start_time = time.time()
        embedding = await self.embed(text)
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embedding=embedding.tolist(),
            dimensions=len(embedding),
            model_name=self.model,
            processing_time_ms=processing_time_ms,
            metadata={"provider": "ollama"}
        )
    
    async def batch_embed(self, texts: List[str]):
        """Generate embeddings for multiple texts (interface compliance)"""
        from globule.core.models import EmbeddingResult
        import time
        
        start_time = time.time()
        embeddings = await self.embed_batch(texts)
        processing_time_ms = (time.time() - start_time) * 1000
        
        results = []
        for embedding in embeddings:
            results.append(EmbeddingResult(
                embedding=embedding.tolist(),
                dimensions=len(embedding),
                model_name=self.model,
                processing_time_ms=processing_time_ms / len(texts),  # Distribute time across batch
                metadata={"provider": "ollama"}
            ))
        return results
    
    def get_dimensions(self) -> int:
        """Interface compliance method"""
        return self.get_dimension()
    
    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            session = await self._get_session()
            
            # Check if Ollama is running
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return False
                
                data = await response.json()
                available_models = [model["name"] for model in data.get("models", [])]
                
                # Check if our model is available, accounting for tags like ":latest"
                return any(m.startswith(self.model) for m in available_models)
                
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False