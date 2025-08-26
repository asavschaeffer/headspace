"""
Contract tests for Phase 4 processor interfaces and models.

These tests verify that the contracts (ABCs and models) are properly defined
and can be implemented correctly. They serve as the foundation for all
processor implementations.
"""
import pytest
import time
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

from globule.core.interfaces import IProcessor
from globule.core.models import (
    GlobuleV1, 
    ProcessedContent, 
    StructuredQuery,
    ProcessedGlobuleV1
)
from globule.core.errors import ParserError


class MockProcessor(IProcessor):
    """Mock processor implementation for contract testing."""
    
    def can_process(self, globule: GlobuleV1) -> float:
        # Mock: handle text content with moderate confidence
        if globule.raw_text and len(globule.raw_text) > 0:
            return 0.7
        return 0.0
    
    async def process(self, globule: GlobuleV1) -> ProcessedContent:
        start_time = time.time()
        
        # Mock processing
        structured_data = {
            "word_count": len(globule.raw_text.split()),
            "character_count": len(globule.raw_text),
            "detected_type": "text"
        }
        
        metadata = {
            "processed_at": datetime.utcnow().isoformat(),
            "globule_id": str(globule.globule_id)
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedContent(
            structured_data=structured_data,
            metadata=metadata,
            confidence=0.85,
            processor_type=self.get_processor_type(),
            processing_time_ms=processing_time
        )
    
    def get_processor_type(self) -> str:
        return "mock_text"


class TestProcessorContracts:
    """Test the IProcessor interface contract."""
    
    def test_processor_interface_methods(self):
        """Verify IProcessor has all required methods."""
        processor = MockProcessor()
        
        # Verify all abstract methods are implemented
        assert hasattr(processor, 'can_process')
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'get_processor_type')
        assert callable(processor.can_process)
        assert callable(processor.process)
        assert callable(processor.get_processor_type)
    
    def test_can_process_returns_float(self):
        """Verify can_process returns valid confidence scores."""
        processor = MockProcessor()
        globule = GlobuleV1(
            raw_text="Test content",
            source="test"
        )
        
        confidence = processor.can_process(globule)
        
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_returns_processed_content(self):
        """Verify process returns valid ProcessedContent."""
        processor = MockProcessor()
        globule = GlobuleV1(
            raw_text="Test content for processing",
            source="test"
        )
        
        result = await processor.process(globule)
        
        assert isinstance(result, ProcessedContent)
        assert isinstance(result.structured_data, dict)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.confidence, (int, float))
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.processor_type, str)
        assert isinstance(result.processing_time_ms, (int, float))
        assert result.processing_time_ms >= 0
    
    def test_get_processor_type_returns_string(self):
        """Verify get_processor_type returns valid string."""
        processor = MockProcessor()
        
        processor_type = processor.get_processor_type()
        
        assert isinstance(processor_type, str)
        assert len(processor_type) > 0


class TestProcessedContentModel:
    """Test the ProcessedContent model contract."""
    
    def test_processed_content_creation(self):
        """Test ProcessedContent model creation and validation."""
        content = ProcessedContent(
            structured_data={"key": "value"},
            metadata={"source": "test"},
            confidence=0.9,
            processor_type="test",
            processing_time_ms=100.5
        )
        
        assert content.structured_data == {"key": "value"}
        assert content.metadata == {"source": "test"}
        assert content.confidence == 0.9
        assert content.processor_type == "test"
        assert content.processing_time_ms == 100.5
    
    def test_processed_content_defaults(self):
        """Test ProcessedContent with default values."""
        content = ProcessedContent(
            confidence=0.8,
            processor_type="test",
            processing_time_ms=50.0
        )
        
        assert content.structured_data == {}
        assert content.metadata == {}
        assert content.confidence == 0.8
    
    def test_processed_content_validation(self):
        """Test ProcessedContent field validation."""
        # Test confidence bounds
        with pytest.raises(ValueError):
            ProcessedContent(
                confidence=-0.1,  # Below 0.0
                processor_type="test",
                processing_time_ms=50.0
            )
        
        with pytest.raises(ValueError):
            ProcessedContent(
                confidence=1.1,  # Above 1.0
                processor_type="test", 
                processing_time_ms=50.0
            )
        
        # Test negative processing time
        with pytest.raises(ValueError):
            ProcessedContent(
                confidence=0.8,
                processor_type="test",
                processing_time_ms=-10.0
            )


class TestStructuredQueryModel:
    """Test the StructuredQuery model contract."""
    
    def test_structured_query_creation(self):
        """Test StructuredQuery model creation."""
        query = StructuredQuery(
            domain="valet",
            filters={"license_plate": "ABC-123"},
            limit=20,
            sort_by="created_at",
            sort_desc=False
        )
        
        assert query.domain == "valet"
        assert query.filters == {"license_plate": "ABC-123"}
        assert query.limit == 20
        assert query.sort_by == "created_at"
        assert query.sort_desc == False
    
    def test_structured_query_defaults(self):
        """Test StructuredQuery with default values."""
        query = StructuredQuery(domain="test")
        
        assert query.domain == "test"
        assert query.filters == {}
        assert query.limit == 10
        assert query.sort_by is None
        assert query.sort_desc == True
    
    def test_structured_query_validation(self):
        """Test StructuredQuery field validation."""
        # Test limit bounds
        with pytest.raises(ValueError):
            StructuredQuery(domain="test", limit=0)  # Below 1
        
        with pytest.raises(ValueError):
            StructuredQuery(domain="test", limit=101)  # Above 100


class TestProcessorIntegration:
    """Integration tests for processor contracts."""
    
    @pytest.mark.asyncio
    async def test_processor_workflow(self):
        """Test complete processor workflow."""
        processor = MockProcessor()
        globule = GlobuleV1(
            raw_text="Integration test content with multiple words",
            source="integration_test"
        )
        
        # Check processing capability
        confidence = processor.can_process(globule)
        assert confidence > 0.5  # Should be able to process
        
        # Process if confident enough
        if confidence > 0.5:
            result = await processor.process(globule)
            
            # Verify result quality
            assert result.confidence > 0.0
            assert result.processor_type == "mock_text"
            assert "word_count" in result.structured_data
            assert result.structured_data["word_count"] > 0
            assert "processed_at" in result.metadata
    
    @pytest.mark.asyncio
    async def test_processor_empty_content(self):
        """Test processor with empty content."""
        processor = MockProcessor()
        globule = GlobuleV1(raw_text="", source="test")
        
        # Should return low confidence for empty content
        confidence = processor.can_process(globule)
        assert confidence == 0.0
        
        # But should still be able to process if forced
        result = await processor.process(globule)
        assert isinstance(result, ProcessedContent)
        assert result.structured_data["word_count"] == 0