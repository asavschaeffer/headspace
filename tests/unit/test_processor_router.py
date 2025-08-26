"""
Unit tests for ProcessorRouter and ProcessorFactory.

Tests the routing logic, processor registration, batch processing,
and dependency injection patterns.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from globule.core.models import GlobuleV1, ProcessedContent
from globule.core.errors import ParserError
from globule.processors.processor_router import ProcessorRouter
from globule.processors.processor_factory import ProcessorFactory


class MockProcessor:
    """Mock processor for testing router functionality."""
    
    def __init__(self, processor_type: str = "mock", confidence: float = 0.8):
        self.processor_type = processor_type
        self.confidence = confidence
        self.process_called = False
    
    def can_process(self, globule: GlobuleV1) -> float:
        return self.confidence
    
    async def process(self, globule: GlobuleV1) -> ProcessedContent:
        self.process_called = True
        
        return ProcessedContent(
            structured_data={
                "title": f"Processed by {self.processor_type}",
                "category": "test",
                "content_type": self.processor_type
            },
            metadata={
                "processor_type": self.processor_type,
                "test_processor": True
            },
            confidence=self.confidence,
            processor_type=self.processor_type,
            processing_time_ms=10.0
        )
    
    def get_processor_type(self) -> str:
        return self.processor_type


class TestProcessorRouter:
    """Test the ProcessorRouter functionality."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = ProcessorRouter()
        
        assert len(router.get_registered_processors()) == 0
        assert isinstance(router.get_processor_capabilities(), dict)
    
    def test_register_processor(self):
        """Test processor registration."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("test")
        
        router.register_processor(mock_processor)
        
        processors = router.get_registered_processors()
        assert len(processors) == 1
        assert processors[0] == mock_processor
    
    @pytest.mark.asyncio
    async def test_route_and_process_no_processors(self):
        """Test routing when no processors are registered."""
        router = ProcessorRouter()
        globule = GlobuleV1(raw_text="test content", source="test")
        
        with pytest.raises(ParserError, match="No processors registered"):
            await router.route_and_process(globule)
    
    @pytest.mark.asyncio
    async def test_route_and_process_single_processor(self):
        """Test routing with single processor."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("test", confidence=0.9)
        router.register_processor(mock_processor)
        
        globule = GlobuleV1(raw_text="test content", source="test")
        result = await router.route_and_process(globule)
        
        assert isinstance(result, ProcessedContent)
        assert result.processor_type == "test"
        assert mock_processor.process_called
        assert result.structured_data["title"] == "Processed by test"
    
    @pytest.mark.asyncio
    async def test_route_and_process_multiple_processors(self):
        """Test routing selects best processor based on confidence."""
        router = ProcessorRouter()
        
        # Register processors with different confidence levels
        low_confidence = MockProcessor("low", confidence=0.3)
        high_confidence = MockProcessor("high", confidence=0.9)
        medium_confidence = MockProcessor("medium", confidence=0.6)
        
        router.register_processor(low_confidence)
        router.register_processor(high_confidence)
        router.register_processor(medium_confidence)
        
        globule = GlobuleV1(raw_text="test content", source="test")
        result = await router.route_and_process(globule)
        
        # Should select the high confidence processor
        assert result.processor_type == "high"
        assert high_confidence.process_called
        assert not low_confidence.process_called
        assert not medium_confidence.process_called
    
    @pytest.mark.asyncio
    async def test_route_and_process_zero_confidence(self):
        """Test routing when all processors have zero confidence."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("test", confidence=0.0)
        router.register_processor(mock_processor)
        
        globule = GlobuleV1(raw_text="test content", source="test")
        
        with pytest.raises(ParserError, match="No processor can handle content"):
            await router.route_and_process(globule)
    
    @pytest.mark.asyncio
    async def test_route_and_process_processor_error(self):
        """Test routing handles processor errors."""
        router = ProcessorRouter()
        
        # Mock processor that raises an error
        mock_processor = MockProcessor("error", confidence=0.8)
        
        async def failing_process(globule):
            raise Exception("Processing failed")
        
        mock_processor.process = failing_process
        router.register_processor(mock_processor)
        
        globule = GlobuleV1(raw_text="test content", source="test")
        
        with pytest.raises(ParserError, match="Processing failed"):
            await router.route_and_process(globule)
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test successful batch processing."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("batch", confidence=0.8)
        router.register_processor(mock_processor)
        
        globules = [
            GlobuleV1(raw_text="content 1", source="test"),
            GlobuleV1(raw_text="content 2", source="test"),
            GlobuleV1(raw_text="content 3", source="test")
        ]
        
        results = await router.process_batch(globules)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ProcessedContent)
            assert result.processor_type == "batch"
    
    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self):
        """Test batch processing handles individual errors."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("batch", confidence=0.8)
        
        # Mock processor that fails on second item
        original_process = mock_processor.process
        call_count = 0
        
        async def selective_failure(globule):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Second item failed")
            return await original_process(globule)
        
        mock_processor.process = selective_failure
        router.register_processor(mock_processor)
        
        globules = [
            GlobuleV1(raw_text="content 1", source="test"),
            GlobuleV1(raw_text="content 2", source="test"),
            GlobuleV1(raw_text="content 3", source="test")
        ]
        
        results = await router.process_batch(globules)
        
        assert len(results) == 3
        assert results[0].processor_type == "batch"
        assert results[1].processor_type == "fallback"  # Error case
        assert results[1].metadata["batch_error"] is True
        assert results[2].processor_type == "batch"
    
    def test_get_processor_capabilities(self):
        """Test processor capabilities reporting."""
        router = ProcessorRouter()
        
        # Register different processor types
        image_processor = MockProcessor("image", confidence=0.8)
        text_processor = MockProcessor("text", confidence=0.7)
        
        router.register_processor(image_processor)
        router.register_processor(text_processor)
        
        capabilities = router.get_processor_capabilities()
        
        assert "image" in capabilities
        assert "text" in capabilities
        assert "content_processing" in capabilities["image"]
        assert "file_path_detection" in capabilities["image"]
        assert "content_processing" in capabilities["text"]
        assert "nlp_parsing" in capabilities["text"]
    
    def test_get_routing_stats(self):
        """Test routing statistics."""
        router = ProcessorRouter()
        mock_processor = MockProcessor("stats", confidence=0.8)
        router.register_processor(mock_processor)
        
        stats = router.get_routing_stats()
        
        assert stats["registered_processors"] == 1
        assert stats["processor_types"] == ["stats"]
        assert stats["router_version"] == "4.0.0"
        assert "capabilities" in stats


class TestProcessorFactory:
    """Test the ProcessorFactory functionality."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        assert factory.embedding_adapter == mock_adapter
        assert factory.config is not None
    
    def test_create_image_processor(self):
        """Test image processor creation."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        processor = factory.create_image_processor()
        
        # Import here to avoid circular dependencies
        from globule.processors.image_processor import ImageProcessor
        assert isinstance(processor, ImageProcessor)
        assert processor.multimodal_adapter == mock_adapter
        assert processor.processor_type == "image"
    
    def test_create_processor_router_default(self):
        """Test processor router creation with defaults."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        router = factory.create_processor_router()
        
        assert isinstance(router, ProcessorRouter)
        processors = router.get_registered_processors()
        assert len(processors) == 1
        assert processors[0].get_processor_type() == "image"
    
    def test_create_processor_router_custom_types(self):
        """Test processor router creation with custom types."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        router = factory.create_processor_router(['image'])
        
        processors = router.get_registered_processors()
        assert len(processors) == 1
        assert processors[0].get_processor_type() == "image"
    
    def test_create_processor_router_unknown_type(self):
        """Test processor router creation with unknown type."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        router = factory.create_processor_router(['unknown'])
        
        processors = router.get_registered_processors()
        assert len(processors) == 0  # Unknown type should be skipped
    
    def test_get_available_processor_types(self):
        """Test getting available processor types."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        types = factory.get_available_processor_types()
        
        assert types == ['image']
    
    def test_create_all_processors(self):
        """Test creating all available processors."""
        mock_adapter = Mock()
        factory = ProcessorFactory(mock_adapter)
        
        processors = factory.create_all_processors()
        
        assert len(processors) == 1
        assert processors[0].get_processor_type() == "image"
    
    def test_get_factory_info(self):
        """Test factory information."""
        mock_adapter = Mock()
        mock_adapter.__class__.__name__ = "MockAdapter"
        factory = ProcessorFactory(mock_adapter)
        
        info = factory.get_factory_info()
        
        assert info["available_processors"] == ['image']
        assert info["embedding_adapter"] == "MockAdapter"
        assert info["factory_version"] == "4.0.0"
        assert info["dependency_injection"] is True


@pytest.mark.asyncio
async def test_router_factory_integration():
    """Integration test for router and factory working together."""
    mock_adapter = Mock()
    factory = ProcessorFactory(mock_adapter)
    
    # Create router with factory
    router = factory.create_processor_router()
    
    # Test routing with real processors
    globule = GlobuleV1(raw_text="/path/to/image.jpg", source="test")
    
    # Should select image processor for image path
    # Note: This would need mocking for the actual processing
    processors = router.get_registered_processors()
    assert len(processors) == 1
    assert processors[0].get_processor_type() == "image"