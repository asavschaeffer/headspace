"""
Embedding Services.

Handles semantic embedding generation for text using various providers.
"""

from .ollama_provider import OllamaEmbeddingProvider
from .ollama_adapter import OllamaEmbeddingAdapter

__all__ = ['OllamaEmbeddingProvider', 'OllamaEmbeddingAdapter']