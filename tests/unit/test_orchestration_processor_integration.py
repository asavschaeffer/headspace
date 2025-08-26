"""
Unit tests for OrchestrationEngine processor integration.

Tests the integration of processor routing within the orchestration workflow,
including structured queries and processor statistics.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from globule.core.models import GlobuleV1, ProcessedGlobuleV1, ProcessedContent, StructuredQuery
from globule.core.errors import ParserError, StorageError
from globule.orchestration.engine import GlobuleOrchestrator


class MockProcessorRouter:
    """Mock processor router for testing orchestration integration."""
    
    def __init__(self, should_succeed: bool = True, confidence: float = 0.8):
        self.should_succeed = should_succeed
        self.confidence = confidence
        self.route_and_process_called = False
    
    async def route_and_process(self, globule: GlobuleV1) -> ProcessedContent:
        self.route_and_process_called = True
        
        if not self.should_succeed:
            raise ParserError("Mock processor routing failed")
        
        return ProcessedContent(
            structured_data={
                "title": "Processed image",
                "category": "media",
                "domain": "image",
                "content_type": "image"
            },
            metadata={
                "processor_type": "image",
                "multimodal_available": True
            },
            confidence=self.confidence,
            processor_type="image",
            processing_time_ms=25.0
        )
    
    def get_routing_stats(self):
        return {
            "registered_processors": 1,
            "processor_types": ["image"],
            "router_version": "4.0.0"
        }
    
    def get_processor_capabilities(self):
        return {
            "image": ["content_processing", "file_path_detection", "exif_extraction"]
        }


class TestOrchestrationProcessorIntegration:
    """Test OrchestrationEngine processor integration."""
    
    @pytest.mark.asyncio
    async def test_orchestration_without_processor_router(self):
        """Test orchestration works without processor router (backward compatibility)."""
        # Setup mocks for existing functionality
        mock_parser = AsyncMock()
        mock_parser.parse.return_value = {"title": "Test", "category": "note"}
        
        mock_embedding = AsyncMock()
        mock_embedding.embed_single.return_value = Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
        
        mock_storage = Mock()
        
        # Create orchestrator without processor router
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage
        )
        
        globule = GlobuleV1(raw_text="test content", source="test")
        result = await orchestrator.process(globule)
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert result.parsed_data["title"] == "Test"
        assert "processor_type" not in result.provider_metadata
    
    @pytest.mark.asyncio
    async def test_orchestration_with_processor_router_success(self):
        """Test orchestration with successful processor routing."""
        # Setup mocks for existing functionality
        mock_parser = AsyncMock()
        mock_parser.parse.return_value = {"title": "Text content", "category": "note"}
        
        mock_embedding = AsyncMock()
        mock_embedding.embed_single.return_value = Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
        
        mock_storage = Mock()
        
        # Setup processor router
        mock_router = MockProcessorRouter(should_succeed=True, confidence=0.9)
        
        # Create orchestrator with processor router
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage,
            processor_router=mock_router
        )
        
        globule = GlobuleV1(raw_text="/path/to/image.jpg", source="test")
        result = await orchestrator.process(globule)
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert mock_router.route_and_process_called
        
        # Should use processor result due to high confidence
        assert result.parsed_data["category"] == "media"
        assert result.parsed_data["domain"] == "image"
        
        # Should include processor metadata
        assert result.provider_metadata["processor_type"] == "image"
        assert result.provider_metadata["processor_confidence"] == 0.9
        assert "processor_time_ms" in result.provider_metadata
    
    @pytest.mark.asyncio
    async def test_orchestration_with_processor_router_low_confidence(self):
        """Test orchestration uses parser result when processor confidence is low."""
        # Setup mocks for existing functionality
        mock_parser = AsyncMock()
        mock_parser.parse.return_value = {"title": "Text content", "category": "note"}
        
        mock_embedding = AsyncMock()
        mock_embedding.embed_single.return_value = Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
        
        mock_storage = Mock()
        
        # Setup processor router with low confidence
        mock_router = MockProcessorRouter(should_succeed=True, confidence=0.3)
        
        # Create orchestrator with processor router
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage,
            processor_router=mock_router
        )
        
        globule = GlobuleV1(raw_text="some text content", source="test")
        result = await orchestrator.process(globule)
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert mock_router.route_and_process_called
        
        # Should use parser result due to low processor confidence
        assert result.parsed_data["category"] == "note"
        assert result.parsed_data["title"] == "Text content"
        
        # Should still include processor metadata
        assert result.provider_metadata["processor_confidence"] == 0.3
    
    @pytest.mark.asyncio
    async def test_orchestration_with_processor_router_failure(self):
        """Test orchestration handles processor router failure gracefully."""
        # Setup mocks for existing functionality
        mock_parser = AsyncMock()
        mock_parser.parse.return_value = {"title": "Text content", "category": "note"}
        
        mock_embedding = AsyncMock()
        mock_embedding.embed_single.return_value = Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
        
        mock_storage = Mock()
        
        # Setup processor router that fails
        mock_router = MockProcessorRouter(should_succeed=False)
        
        # Create orchestrator with processor router
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage,
            processor_router=mock_router
        )
        
        globule = GlobuleV1(raw_text="/path/to/image.jpg", source="test")
        result = await orchestrator.process(globule)
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert mock_router.route_and_process_called
        
        # Should fall back to parser result
        assert result.parsed_data["category"] == "note"
        assert result.parsed_data["title"] == "Text content"
        
        # Should not include processor metadata due to failure
        assert "processor_type" not in result.provider_metadata
    
    @pytest.mark.asyncio
    async def test_query_structured_success(self):
        """Test structured query execution."""
        mock_parser = Mock()
        mock_embedding = Mock()
        mock_storage = AsyncMock()
        
        # Mock storage manager response
        expected_results = [
            Mock(spec=ProcessedGlobuleV1),
            Mock(spec=ProcessedGlobuleV1)
        ]
        mock_storage.query_structured.return_value = expected_results
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage
        )
        
        query = StructuredQuery(
            domain="image",
            filters={"category": "media"},
            limit=10
        )
        
        results = await orchestrator.query_structured(query)
        
        assert results == expected_results
        mock_storage.query_structured.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_query_structured_storage_error(self):
        """Test structured query handles storage errors."""
        mock_parser = Mock()
        mock_embedding = Mock()
        mock_storage = AsyncMock()
        
        # Mock storage manager to raise error
        mock_storage.query_structured.side_effect = StorageError("Query failed")
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage
        )
        
        query = StructuredQuery(
            domain="image",
            filters={"category": "media"},
            limit=10
        )
        
        results = await orchestrator.query_structured(query)
        
        assert results == []  # Should return empty list on error
        mock_storage.query_structured.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_get_processor_stats_without_router(self):
        """Test processor stats when no router is configured."""
        mock_parser = Mock()
        mock_embedding = Mock()
        mock_storage = Mock()
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage
        )
        
        stats = await orchestrator.get_processor_stats()
        
        assert stats == {"processors_enabled": False}
    
    @pytest.mark.asyncio
    async def test_get_processor_stats_with_router(self):
        """Test processor stats when router is configured."""
        mock_parser = Mock()
        mock_embedding = Mock()
        mock_storage = Mock()
        mock_router = MockProcessorRouter()
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage,
            processor_router=mock_router
        )
        
        stats = await orchestrator.get_processor_stats()
        
        assert stats["processors_enabled"] is True
        assert "routing_stats" in stats
        assert "capabilities" in stats
        assert stats["routing_stats"]["registered_processors"] == 1
        assert stats["capabilities"]["image"] == ["content_processing", "file_path_detection", "exif_extraction"]
    
    @pytest.mark.asyncio 
    async def test_capture_thought_with_processor_router(self):
        """Test capture_thought uses processor routing when available."""
        # Setup mocks
        mock_parser = AsyncMock()
        mock_parser.parse.return_value = {"title": "Text", "category": "note"}
        
        mock_embedding = AsyncMock()
        mock_embedding.embed_single.return_value = Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
        
        mock_storage = Mock()
        mock_router = MockProcessorRouter(confidence=0.9)
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=mock_parser,
            embedding_provider=mock_embedding,
            storage_manager=mock_storage,
            processor_router=mock_router
        )
        
        result = await orchestrator.capture_thought("/path/to/image.jpg", source="api")
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert mock_router.route_and_process_called
        assert result.parsed_data["category"] == "media"  # From processor
        mock_storage.save.assert_called_once()


@pytest.mark.asyncio
async def test_orchestration_concurrent_processing():
    """Test that orchestration processes embedding, parsing, and routing concurrently."""
    # Setup mocks with delays to verify concurrency
    mock_parser = AsyncMock()
    mock_embedding = AsyncMock()
    mock_storage = Mock()
    mock_router = Mock()
    
    # Add delays to verify concurrent execution
    async def slow_parse(text):
        await asyncio.sleep(0.01)
        return {"title": "Parsed", "category": "note"}
    
    async def slow_embed(text):
        await asyncio.sleep(0.01)
        return Mock(embedding=[0.1, 0.2], processing_time_ms=10.0)
    
    async def slow_route(globule):
        await asyncio.sleep(0.01)
        return ProcessedContent(
            structured_data={"title": "Routed", "category": "media"},
            metadata={},
            confidence=0.8,
            processor_type="image",
            processing_time_ms=15.0
        )
    
    mock_parser.parse.side_effect = slow_parse
    mock_embedding.embed_single.side_effect = slow_embed
    mock_router.route_and_process = slow_route
    
    orchestrator = GlobuleOrchestrator(
        parser_provider=mock_parser,
        embedding_provider=mock_embedding,
        storage_manager=mock_storage,
        processor_router=mock_router
    )
    
    start_time = asyncio.get_event_loop().time()
    globule = GlobuleV1(raw_text="/path/to/image.jpg", source="test")
    result = await orchestrator.process(globule)
    end_time = asyncio.get_event_loop().time()
    
    # Should take ~0.01 seconds (concurrent) rather than ~0.03 (sequential)
    assert (end_time - start_time) < 0.025
    assert isinstance(result, ProcessedGlobuleV1)