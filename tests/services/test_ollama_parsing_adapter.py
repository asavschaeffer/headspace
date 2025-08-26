import unittest
from unittest.mock import AsyncMock

from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.core.errors import ParserError
from globule.services.parsing.ollama_parser import OllamaParser


class TestOllamaParsingAdapter(unittest.TestCase):
    async def test_parse_success(self):
        # Arrange
        mock_provider = AsyncMock(spec=OllamaParser)
        mock_provider.parse = AsyncMock(return_value={"title": "test", "category": "note"})
        
        # Act
        adapter = OllamaParsingAdapter(mock_provider)
        result = await adapter.parse("test text")
        
        # Assert
        mock_provider.parse.assert_awaited_once_with("test text")
        self.assertEqual(result, {"title": "test", "category": "note"})
    
    async def test_parse_failure_raises_parser_error(self):
        # Arrange
        mock_provider = AsyncMock(spec=OllamaParser)
        mock_provider.parse.side_effect = Exception("Provider exploded")
        
        # Act & Assert
        adapter = OllamaParsingAdapter(mock_provider)
        async with self.assertRaises(ParserError):
            await adapter.parse("test text")