"""
Unit tests for Phase 4 parsing processors.

Tests the processor adapter pattern and specific implementations including
ImageProcessor and TextProcessor functionality.
"""
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from globule.core.models import GlobuleV1, ProcessedContent
from globule.core.errors import ParserError
from globule.processors.processor_adapter import ProcessorAdapter
from globule.processors.image_processor import ImageProcessor


class MockProcessor(ProcessorAdapter):
    """Mock processor for testing base adapter functionality."""
    
    def __init__(self, processor_type: str = "mock", confidence: float = 0.8):
        super().__init__(processor_type)
        self.mock_confidence = confidence
        self.process_called = False
        self.should_raise = False
    
    def _calculate_processing_confidence(self, globule: GlobuleV1) -> float:
        return self.mock_confidence
    
    async def _process_content(self, globule: GlobuleV1) -> tuple[dict, dict]:
        self.process_called = True
        
        if self.should_raise:
            raise Exception("Mock processing error")
        
        structured_data = {
            "title": "Mock processed content",
            "category": "test",
            "content_type": "mock",
            "word_count": len(globule.raw_text.split()),
            "confidence_score": 0.9
        }
        
        metadata = {
            "mock_processor": True,
            "test_run": True
        }
        
        return structured_data, metadata


class TestProcessorAdapter:
    """Test the base ProcessorAdapter functionality."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization with type."""
        processor = MockProcessor("test_type")
        
        assert processor.processor_type == "test_type"
        assert processor.get_processor_type() == "test_type"
        assert processor.config is not None
    
    def test_can_process_empty_content(self):
        """Test can_process with empty content."""
        processor = MockProcessor()
        
        # Empty text
        globule = GlobuleV1(raw_text="", source="test")
        assert processor.can_process(globule) == 0.0
        
        
        # Whitespace only
        globule = GlobuleV1(raw_text="   \n  \t  ", source="test")
        assert processor.can_process(globule) == 0.0
    
    def test_can_process_valid_content(self):
        """Test can_process with valid content."""
        processor = MockProcessor(confidence=0.75)
        globule = GlobuleV1(raw_text="Valid test content", source="test")
        
        confidence = processor.can_process(globule)
        assert confidence == 0.75
    
    def test_can_process_error_handling(self):
        """Test can_process error handling."""
        processor = MockProcessor()
        
        # Mock the confidence calculation to raise an error
        with patch.object(processor, '_calculate_processing_confidence', side_effect=Exception("Test error")):
            globule = GlobuleV1(raw_text="Test content", source="test")
            confidence = processor.can_process(globule)
            assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_process_successful(self):
        """Test successful content processing."""
        processor = MockProcessor()
        globule = GlobuleV1(raw_text="Test content for processing", source="test")
        
        result = await processor.process(globule)
        
        assert isinstance(result, ProcessedContent)
        assert result.processor_type == "mock"
        assert result.confidence > 0.0
        assert result.processing_time_ms >= 0
        assert processor.process_called
        
        # Check structured data
        assert "title" in result.structured_data
        assert "category" in result.structured_data
        assert result.structured_data["word_count"] == 4
        
        # Check metadata
        assert "mock_processor" in result.metadata
        assert "processed_at" in result.metadata
        assert "globule_id" in result.metadata
    
    @pytest.mark.asyncio
    async def test_process_error_handling(self):
        """Test process error handling and fallback."""
        processor = MockProcessor()
        processor.should_raise = True
        
        globule = GlobuleV1(raw_text="Test content", source="test")
        result = await processor.process(globule)
        
        # Should return fallback result, not raise exception
        assert isinstance(result, ProcessedContent)
        assert result.confidence == 0.1  # Low confidence for fallback
        assert "error" in result.structured_data
        assert result.metadata["processor_error"] is True
        assert result.metadata["fallback_result"] is True
    
    @pytest.mark.asyncio
    async def test_process_validation_error(self):
        """Test input validation errors."""
        processor = MockProcessor()
        
        # Invalid globule type - should return fallback, not raise
        result = await processor.process("not a globule")
        assert result.metadata["fallback_result"] is True
        assert result.metadata["processor_error"] is True
        
        # Empty raw_text
        globule = GlobuleV1(raw_text="", source="test")
        result = await processor.process(globule)
        # Should handle gracefully and return fallback
        assert result.metadata["fallback_result"] is True
    
    def test_content_profiling(self):
        """Test content profiling functionality."""
        processor = MockProcessor()
        
        # Simple text
        profile = processor._extract_content_profile("Hello world")
        assert profile["length"] == 11
        assert profile["word_count"] == 2
        assert profile["line_count"] == 1
        assert not profile["has_urls"]
        assert not profile["has_code"]
        
        # Code content
        code_text = "def hello():\n    print('world')"
        profile = processor._extract_content_profile(code_text)
        assert profile["has_code"]
        assert profile["technical_score"] > 0.5
        
        # URL content
        url_text = "Check out https://example.com for more info"
        profile = processor._extract_content_profile(url_text)
        assert profile["has_urls"]
        assert profile["technical_score"] > 0.0




class TestImageProcessor:
    """Test the ImageProcessor implementation."""
    
    def test_image_processor_initialization(self):
        """Test ImageProcessor initialization."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        assert processor.processor_type == "image"
        assert processor.cache_dir.exists()
        assert isinstance(processor.IMAGE_EXTENSIONS, set)
        assert isinstance(processor.IMAGE_MIME_TYPES, set)
        assert processor.multimodal_adapter == mock_adapter
    
    def test_image_content_detection_file_path(self):
        """Test image content detection for file paths."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # Existing image file
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            globule = GlobuleV1(raw_text=tmp_file.name, source="test")
            confidence = processor._calculate_processing_confidence(globule)
            assert confidence == 0.95
        
        # Non-existing image file
        globule = GlobuleV1(raw_text="/path/to/image.png", source="test")
        confidence = processor._calculate_processing_confidence(globule)
        assert confidence == 0.8
        
        # Non-image file
        globule = GlobuleV1(raw_text="/path/to/document.txt", source="test")
        confidence = processor._calculate_processing_confidence(globule)
        assert confidence == 0.0
    
    def test_image_url_detection(self):
        """Test image URL detection."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # Valid image URL
        assert processor._is_image_url("https://example.com/image.jpg")
        assert processor._is_image_url("http://site.com/photo.png")
        
        # Invalid URLs
        assert not processor._is_image_url("https://example.com/page.html")
        assert not processor._is_image_url("not-a-url")
        assert not processor._is_image_url("ftp://example.com/image.jpg")
    
    def test_base64_image_detection(self):
        """Test base64 image detection."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # Data URL format
        assert processor._is_base64_image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
        
        # Pure base64 (simplified check)
        long_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert processor._is_base64_image(long_base64)
        
        # Not base64
        assert not processor._is_base64_image("regular text content")
        assert not processor._is_base64_image("short")
    
    def test_image_confidence_with_context(self):
        """Test image confidence with context hints."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # MIME type in context
        globule = GlobuleV1(
            raw_text="some content",
            source="test",
            initial_context={"content_type": "image/jpeg"}
        )
        confidence = processor._calculate_processing_confidence(globule)
        assert confidence == 0.9
        
        # Image keywords in text
        globule = GlobuleV1(raw_text="This is an image of a sunset", source="test")
        confidence = processor._calculate_processing_confidence(globule)
        assert confidence == 0.3
    
    @pytest.mark.asyncio
    async def test_image_processing_file(self):
        """Test image processing for file paths."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        processor.multimodal_available = False  # Use fallback for testing
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name
        
        try:
            globule = GlobuleV1(raw_text=tmp_path, source="test")
            
            with patch.object(processor, '_extract_exif_metadata', return_value={"exif": {"width": 100}}), \
                 patch.object(processor, '_generate_thumbnail', return_value=None):
                
                structured_data, metadata = await processor._process_content(globule)
                
                # Verify structure
                assert structured_data["category"] == "media"
                assert structured_data["domain"] == "image"
                assert structured_data["content_type"] == "image"
                assert structured_data["technical_metadata"]["source_type"] == "file"
                assert structured_data["technical_metadata"]["file_path"] == tmp_path
                
                # Verify metadata
                assert metadata["source_type"] == "file"
                assert "multimodal_model" in metadata
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_image_processing_error_handling(self):
        """Test image processing error handling."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # Non-existent file
        globule = GlobuleV1(raw_text="/nonexistent/image.jpg", source="test")
        
        with pytest.raises(ParserError, match="Unable to process image source"):
            await processor._process_content(globule)
    
    def test_title_generation(self):
        """Test image title generation."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        # From description
        desc_result = {"description": "A beautiful sunset"}
        title = processor._generate_title(desc_result, "test.jpg")
        assert title == "A beautiful sunset"
        
        # From filename
        desc_result = {"description": "Long description that exceeds the title length limit for proper display"}
        title = processor._generate_title(desc_result, "/path/to/beach_sunset_photo.jpg")
        assert title == "Beach Sunset Photo"
        
        # From URL
        desc_result = {}
        title = processor._generate_title(desc_result, "https://example.com/vacation/photo.jpg")
        assert title == "photo" or title == "Photo"
    
    def test_keyword_extraction(self):
        """Test keyword extraction from image analysis."""
        mock_adapter = Mock()
        processor = ImageProcessor(mock_adapter)
        
        analysis_result = {
            "objects": ["car", "tree", "building"],
            "scene": "urban street with traffic",
            "colors": ["blue", "green", "gray"]
        }
        
        keywords = processor._extract_keywords(analysis_result)
        
        # Should include objects, scene words, colors, and 'image'
        assert "car" in keywords
        assert "tree" in keywords
        assert "blue" in keywords
        assert "image" in keywords
        
        # Should limit and deduplicate
        assert len(keywords) <= 10
        assert len(keywords) == len(set(keywords))  # No duplicates


@pytest.mark.asyncio
async def test_processor_integration():
    """Integration test for processor workflow."""
    # Test image processor
    mock_adapter = Mock()
    image_processor = ImageProcessor(mock_adapter)
    image_processor.multimodal_available = False
    
    # Create test file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_file.write(b"fake image data")
        tmp_path = tmp_file.name
    
    try:
        globule = GlobuleV1(raw_text=tmp_path, source="integration_test")
        
        # Check can process
        confidence = image_processor.can_process(globule)
        assert confidence > 0.9
        
        # Process content with mocked methods
        with patch.object(image_processor, '_extract_exif_metadata', return_value={}), \
             patch.object(image_processor, '_generate_thumbnail', return_value=None):
            
            result = await image_processor.process(globule)
            assert isinstance(result, ProcessedContent)
            assert result.processor_type == "image"
            assert result.structured_data["category"] == "media"
            
    finally:
        Path(tmp_path).unlink(missing_ok=True)