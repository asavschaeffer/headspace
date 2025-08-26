import unittest
import numpy as np
from unittest.mock import AsyncMock

from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter
from globule.core.errors import EmbeddingError
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider


class TestOllamaEmbeddingAdapter(unittest.TestCase):
    async def test_embed_success(self):
        # Arrange
        mock_provider = AsyncMock(spec=OllamaEmbeddingProvider)
        mock_provider.embed = AsyncMock(return_value=np.array([0.1, 0.2]))
        
        # Act
        adapter = OllamaEmbeddingAdapter(mock_provider)
        result = await adapter.embed("test text")
        
        # Assert
        mock_provider.embed.assert_awaited_once_with("test text")
        self.assertEqual(result, [0.1, 0.2])
    
    async def test_embed_failure_raises_embedding_error(self):
        # Arrange
        mock_provider = AsyncMock(spec=OllamaEmbeddingProvider)
        mock_provider.embed.side_effect = Exception("Provider exploded")
        
        # Act & Assert
        adapter = OllamaEmbeddingAdapter(mock_provider)
        async with self.assertRaises(EmbeddingError):
            await adapter.embed("test text")