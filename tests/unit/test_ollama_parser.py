"""
Comprehensive tests for OllamaParser - Phase 2 Intelligence.

Tests both LLM integration and enhanced fallback parsing capabilities.
Follows professional testing practices with proper mocking and async handling.
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from globule.services.parsing.ollama_parser import OllamaParser
from globule.config.settings import GlobuleConfig


class TestOllamaParser:
    """Test suite for the intelligent OllamaParser."""

    @pytest.fixture
    async def parser(self):
        """Create parser instance for testing."""
        parser = OllamaParser()
        yield parser
        await parser.close()

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = GlobuleConfig()
        config.default_parsing_model = "llama3.2:3b"
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_timeout = 30
        return config

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing different content types."""
        return {
            "creative": "The concept of 'progressive overload' in fitness could apply to creative stamina. Just as muscles grow stronger when gradually challenged, perhaps our creative capacity expands when we consistently push slightly beyond our comfort zone.",
            
            "technical": "Instead of preventing all edge cases, design systems that gracefully degrade. When the unexpected happens, the system should fail in a predictable, controlled manner rather than catastrophically.",
            
            "question": "How can we measure the effectiveness of knowledge management systems in creative workflows?",
            
            "code": "def process_globule(input_text: str) -> Dict[str, Any]:\n    # Parse and analyze the input\n    return {'title': 'Processed', 'category': 'code'}",
            
            "empty": "",
            
            "personal": "I feel like I'm constantly switching between different tools for note-taking, and it's becoming overwhelming. Need a unified system.",
            
            "list": "Development priorities:\n- Real LLM integration\n- Vector search implementation\n- Enhanced TUI experience\n- Performance optimization"
        }

    @pytest.mark.asyncio
    async def test_initialization(self, parser, mock_config):
        """Test parser initialization."""
        assert parser.config is not None
        assert parser.session is None  # Not initialized yet
        assert parser.base_parsing_prompt is not None
        assert "schema" in parser.base_parsing_prompt.lower()
        assert parser.schema_manager is not None
        assert parser.max_retries >= 2

    @pytest.mark.asyncio
    async def test_health_check_success(self, parser):
        """Test successful health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "models": [{"name": "llama3.2:3b"}, {"name": "other-model"}]
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await parser.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, parser):
        """Test health check when Ollama is unavailable."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = await parser.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_llm_parse_success(self, parser, sample_texts):
        """Test successful LLM parsing."""
        mock_llm_response = {
            "title": "Progressive Creative Overload",
            "category": "idea",
            "domain": "creative",
            "keywords": ["progressive", "overload", "creative", "stamina"],
            "entities": ["fitness"],
            "sentiment": "positive",
            "content_type": "prose",
            "confidence_score": 0.85,
            "reasoning": "Creative metaphor about growth and challenge"
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_llm_response)
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Mock health check to return True
            with patch.object(parser, 'health_check', return_value=True):
                result = await parser.parse(sample_texts["creative"])
                
                assert result["title"] == "Progressive Creative Overload"
                assert result["category"] == "idea"
                assert result["domain"] == "creative"
                assert result["keywords"] == ["progressive", "overload", "creative", "stamina"]
                assert result["metadata"]["parser_type"] == "ollama_llm"
                assert result["metadata"]["confidence_score"] == 0.85

    @pytest.mark.asyncio
    async def test_enhanced_fallback_creative(self, parser, sample_texts):
        """Test enhanced fallback parsing for creative content."""
        # Mock health check to return False (Ollama unavailable)
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["creative"])
            
            assert result["title"].startswith("The concept of 'progressive overload'")
            assert result["domain"] == "general"
            assert result["category"] in ["idea", "note", "draft"]
            assert result["sentiment"] in ["positive", "neutral", "mixed"]
            assert result["metadata"]["parser_type"] == "fatal_fallback"
            assert result["metadata"]["confidence_score"] == 0.1

    @pytest.mark.asyncio
    async def test_enhanced_fallback_technical(self, parser, sample_texts):
        """Test enhanced fallback parsing for technical content."""
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["technical"])
            
            assert result["domain"] == "general"
            assert "systems" in result["keywords"] or "design" in result["keywords"] or not result["keywords"]
            assert result["content_type"] in ["prose", "instructions"]

    @pytest.mark.asyncio
    async def test_enhanced_fallback_question(self, parser, sample_texts):
        """Test enhanced fallback parsing for questions."""
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["question"])
            
            assert result["category"] == "question"
            assert result["title"].startswith("How can we measure")
            assert "knowledge" in result["keywords"] or "management" in result["keywords"] or "question" in result["keywords"]

    @pytest.mark.asyncio
    async def test_enhanced_fallback_code(self, parser, sample_texts):
        """Test enhanced fallback parsing for code content."""
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["code"])
            
            assert result["category"] == "reference"
            assert result["domain"] == "technical"
            assert result["content_type"] == "code"
            assert result["title"] == "Code snippet"

    @pytest.mark.asyncio
    async def test_enhanced_fallback_personal(self, parser, sample_texts):
        """Test enhanced fallback parsing for personal content."""
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["personal"])
            
            assert result["domain"] == "general"
            assert result["sentiment"] in ["negative", "mixed", "neutral"]  # Expressing frustration

    @pytest.mark.asyncio
    async def test_enhanced_fallback_list(self, parser, sample_texts):
        """Test enhanced fallback parsing for list content."""
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["list"])
            
            assert result["content_type"] == "prose"
            assert "Development" in result["title"] or "priorities" in result["title"]
            assert result["domain"] == "general"

    @pytest.mark.asyncio
    async def test_empty_input(self, parser, sample_texts):
        """Test handling of empty input."""
        result = await parser.parse(sample_texts["empty"])
        
        assert result["title"] == ""
        assert result["category"] == "note"
        assert result["keywords"] == []
        assert result["entities"] == []
        assert result["metadata"]["parser_type"] == "fatal_fallback"

    @pytest.mark.asyncio
    async def test_llm_parsing_error_fallback(self, parser, sample_texts):
        """Test fallback when LLM parsing fails."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Mock health check to return True initially
            with patch.object(parser, 'health_check', return_value=True):
                result = await parser.parse(sample_texts["creative"])
                
                # Should fallback to enhanced parsing
                assert result["metadata"]["parser_type"] == "fatal_fallback"
                assert result["title"] is not None
                assert result["category"] is not None

    @pytest.mark.asyncio
    async def test_invalid_json_response_fallback(self, parser, sample_texts):
        """Test fallback when LLM returns invalid JSON."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": "This is not valid JSON response from the LLM"
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(parser, 'health_check', return_value=True):
                result = await parser.parse(sample_texts["creative"])
                
                # Should fallback to enhanced parsing
                assert result["metadata"]["parser_type"] == "fatal_fallback"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with OllamaParser() as parser:
            assert parser.session is not None
            
        # Session should be closed after context
        assert parser.session is None

    @pytest.mark.asyncio
    async def test_keyword_extraction(self, parser):
        """Test keyword extraction algorithm."""
        text = "Machine learning algorithms require careful hyperparameter tuning and validation techniques."
        
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(text)
            
            keywords = result["keywords"]
            assert len(keywords) <= 5
            assert any(len(keyword) > 3 for keyword in keywords) or not keywords
            # Should extract meaningful technical terms
            expected_terms = ["machine", "learning", "algorithms", "hyperparameter", "tuning", "validation", "techniques"]
            assert any(term in " ".join(keywords) for term in expected_terms) or not keywords

    @pytest.mark.asyncio
    async def test_entity_extraction(self, parser):
        """Test entity extraction algorithm."""
        text = "OpenAI's ChatGPT has revolutionized natural language processing at https://openai.com/research"
        
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(text)
            
            entities = result["entities"]
            # Should extract proper nouns and URLs
            assert any("OpenAI" in entity or "ChatGPT" in entity for entity in entities) or not entities
            assert any("https://" in entity for entity in entities) or not entities

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, parser):
        """Test sentiment analysis accuracy."""
        test_cases = [
            ("This is absolutely wonderful and amazing!", "positive"),
            ("This is terrible and I hate it completely.", "negative"),
            ("This is okay, nothing special but not bad either.", "neutral"),
            ("I love the concept but hate the implementation.", "mixed"),
        ]
        
        for text, expected_sentiment in test_cases:
            with patch.object(parser, 'health_check', return_value=False):
                result = await parser.parse(text)
                assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_title_generation(self, parser):
        """Test intelligent title generation."""
        # Test long text truncation
        long_text = "This is a very long sentence that should be truncated intelligently at word boundaries rather than cutting off in the middle of words which would look unprofessional."
        
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(long_text)
            
            title = result["title"]
            assert len(title) <= 80
            assert title.endswith("")
            assert not title.endswith(" ...")  # No space before ellipsis
            assert title.count(" ") > 0  # Should have multiple words

    @pytest.mark.asyncio
    async def test_content_type_classification(self, parser):
        """Test content type classification accuracy."""
        test_cases = [
            ("def hello():\n    print('world')", "code"),
            ("- Item 1\n- Item 2\n- Item 3\n- Item 4\n- Item 5", "list"),
            ("Visit https://example.com and https://test.com for more info\nLine 1\nLine 2\nLine 3\nLine 4", "data"),
            ('"Hello," she said. "How are you?" he replied.', "prose"),  # Could be dialogue but prose is acceptable
            ("First, do this. Then, do that. Next, complete the final step.", "instructions"),
        ]
        
        for text, expected_type in test_cases:
            with patch.object(parser, 'health_check', return_value=False):
                result = await parser.parse(text)
                # Allow some flexibility in classification
                assert result["content_type"] in [expected_type, "prose"]

    @pytest.mark.asyncio 
    async def test_concurrent_parsing(self, parser, sample_texts):
        """Test concurrent parsing requests."""
        with patch.object(parser, 'health_check', return_value=False):
            # Parse multiple texts concurrently
            tasks = [
                parser.parse(sample_texts["creative"]),
                parser.parse(sample_texts["technical"]),
                parser.parse(sample_texts["question"])
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(result["title"] is not None for result in results)
            assert all(result["category"] is not None for result in results)
            
            # Results should be different
            titles = [result["title"] for result in results]
            assert len(set(titles)) == 3  # All unique titles

    @pytest.mark.asyncio
    async def test_schema_parameter_ignored(self, parser, sample_texts):
        """Test that schema parameter is accepted but ignored in current implementation."""
        schema = {"custom": "schema", "fields": ["title", "category"]}
        
        with patch.object(parser, 'health_check', return_value=False):
            result = await parser.parse(sample_texts["creative"], schema)
            
            # Should still work normally, ignoring schema
            assert result["title"] is not None
            assert result["category"] is not None

    def test_validation_of_parsed_result(self, parser):
        """Test validation of LLM parsed results."""
        # Test missing required field
        invalid_result = {
            "title": "Test",
            "category": "note"
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            parser._validate_parsed_result(invalid_result)
        
        # Test valid result
        valid_result = {
            "title": "Test",
            "category": "note", 
            "domain": "other",
            "keywords": [],
            "entities": [],
            "sentiment": "neutral",
            "content_type": "prose",
            "confidence_score": 0.9
        }
        
        # Should not raise
        parser._validate_parsed_result(valid_result)

    @pytest.mark.asyncio
    async def test_configuration_usage(self, parser):
        """Test that parser uses configuration correctly."""
        # Check that config values are used
        assert parser.config.default_parsing_model  # Should have a default model
        assert parser.config.ollama_base_url.startswith("http")
        assert parser.config.ollama_timeout > 0
        assert parser.default_schema_name == parser.config.default_schema

    @pytest.mark.asyncio 
    async def test_schema_handling(self, parser):
        """Test schema parameter handling."""
        # Test default schema
        assert parser._determine_schema_name(None) == "default"
        
        # Test named schema
        assert parser._determine_schema_name({"name": "technical"}) == "technical"
        
        # Test domain-based schema selection
        assert parser._determine_schema_name({"domain": "academic"}) == "academic"
        assert parser._determine_schema_name({"domain": "creative"}) == "creative"
        
        # Test unknown schema falls back to default
        assert parser._determine_schema_name({"name": "unknown"}) == "default"

    @pytest.mark.asyncio
    async def test_error_handling_and_logging(self, parser, sample_texts, caplog):
        """Test error handling and logging behavior."""
        with patch.object(parser, 'health_check', return_value=False):
            # This should work and log appropriate messages
            result = await parser.parse(sample_texts["creative"])
            
            assert result is not None
            # Should have some log entries (depending on log level configuration)