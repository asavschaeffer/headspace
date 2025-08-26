"""
Code walkthrough test to understand the flow with different content types.

This test creates a realistic scenario with multiple file types and traces
the complete processing flow to understand how the modular architecture works.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import asyncio

from globule.core.api import GlobuleAPI
from globule.orchestration.engine import GlobuleOrchestrator
from globule.processors.processor_router import ProcessorRouter
from globule.storage.sqlite_manager import SQLiteStorageManager


class MockTextProcessor:
    """Mock processor for text content."""
    
    def get_processor_type(self):
        return "text_processor"
    
    def can_process(self, globule):
        # High confidence for text files
        if hasattr(globule, 'initial_context'):
            ext = globule.initial_context.get('file_extension', '')
            if ext in ['.md', '.txt']:
                return 0.9
        return 0.1
    
    async def process(self, globule):
        from globule.core.models import ProcessedContent
        return ProcessedContent(
            structured_data={
                "content_type": "text",
                "title": f"Text from {globule.initial_context.get('filename', 'unknown')}",
                "summary": globule.raw_text[:100] + "...",
                "word_count": len(globule.raw_text.split())
            },
            metadata={"processor": "text_processor"},
            confidence=0.9,
            processor_type="text_processor",
            processing_time_ms=50.0
        )


class MockImageProcessor:
    """Mock processor for image content."""
    
    def get_processor_type(self):
        return "image_processor"
    
    def can_process(self, globule):
        # High confidence for image files
        if hasattr(globule, 'initial_context'):
            ext = globule.initial_context.get('file_extension', '')
            if ext in ['.jpg', '.jpeg', '.png', '.gif']:
                return 0.95
        return 0.0
    
    async def process(self, globule):
        from globule.core.models import ProcessedContent
        return ProcessedContent(
            structured_data={
                "content_type": "image",
                "filename": globule.initial_context.get('filename', 'unknown'),
                "description": "A sample image file",
                "dimensions": "1920x1080",  # Mock dimensions
                "format": "JPEG"
            },
            metadata={
                "processor": "image_processor",
                "has_exif": True
            },
            confidence=0.95,
            processor_type="image_processor",
            processing_time_ms=120.0
        )


class MockPDFProcessor:
    """Mock processor for PDF content."""
    
    def get_processor_type(self):
        return "pdf_processor"
    
    def can_process(self, globule):
        # High confidence for PDF files
        if hasattr(globule, 'initial_context'):
            ext = globule.initial_context.get('file_extension', '')
            if ext == '.pdf':
                return 0.85
        return 0.0
    
    async def process(self, globule):
        from globule.core.models import ProcessedContent
        return ProcessedContent(
            structured_data={
                "content_type": "pdf",
                "filename": globule.initial_context.get('filename', 'unknown'),
                "extracted_text": "This is extracted text from the PDF document...",
                "page_count": 5,
                "has_images": True
            },
            metadata={
                "processor": "pdf_processor",
                "extraction_method": "text_extraction"
            },
            confidence=0.85,
            processor_type="pdf_processor",
            processing_time_ms=200.0
        )


class TestCodeWalkthrough:
    """Walk through the complete indexing flow with different content types."""
    
    @pytest.fixture
    async def test_content_dir(self):
        """Create test directory with different file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "notes.md").write_text("""
# Project Notes

This is a markdown file containing project documentation.
It has multiple lines and structured content.

## Key Points
- Feature A is complete
- Feature B needs testing
- Documentation is in progress
""")
            
            (temp_path / "readme.txt").write_text("""
Project README

This is a simple text file with project information.
It contains plain text without any special formatting.
The content is straightforward and easy to read.
""")
            
            # Create mock binary files (we'll simulate their content)
            (temp_path / "diagram.jpg").write_bytes(b"fake_jpeg_content_for_testing")
            (temp_path / "manual.pdf").write_bytes(b"fake_pdf_content_for_testing")
            
            yield temp_path
    
    @pytest.fixture
    async def mock_orchestrator_with_processors(self):
        """Create orchestrator with mock processors."""
        # Create processor router with mock processors
        router = ProcessorRouter()
        router.register_processor(MockTextProcessor())
        router.register_processor(MockImageProcessor()) 
        router.register_processor(MockPDFProcessor())
        
        # Create mock embedding and parsing providers
        embedding_provider = AsyncMock()
        embedding_provider.embed_single = AsyncMock(return_value=MagicMock(
            embedding=[0.1] * 512,
            processing_time_ms=30.0
        ))
        
        parser_provider = AsyncMock()
        parser_provider.parse = AsyncMock(return_value={"domain": "general", "category": "note"})
        
        storage_manager = AsyncMock()
        
        # Create orchestrator
        orchestrator = GlobuleOrchestrator(
            parser_provider=parser_provider,
            embedding_provider=embedding_provider,
            storage_manager=storage_manager,
            processor_router=router
        )
        
        return orchestrator
    
    @pytest.fixture
    async def mock_storage(self):
        """Create mock storage manager."""
        storage = AsyncMock()
        storage.store_globule_indexed = AsyncMock()
        return storage
    
    @pytest.mark.asyncio
    async def test_complete_indexing_walkthrough(self, test_content_dir, mock_orchestrator_with_processors, mock_storage):
        """Walk through the complete indexing process step by step."""
        
        print("\n" + "="*60)
        print("STARTING INDEXING WALKTHROUGH")
        print("="*60)
        
        # Step 1: Create API instance
        api = GlobuleAPI(storage=mock_storage, orchestrator=mock_orchestrator_with_processors)
        
        print(f"\nTest directory contents:")
        for file in test_content_dir.iterdir():
            print(f"   - {file.name} ({file.suffix})")
        
        # Step 2: Call index_path
        print(f"\nCalling api.index_path('{test_content_dir}')")
        stats = await api.index_path(str(test_content_dir))
        
        # Step 3: Analyze results
        print(f"\nIndexing Results:")
        print(f"   - Files processed: {stats['files_processed']}")
        print(f"   - Files indexed: {stats['files_indexed']}")
        print(f"   - Files skipped: {stats['files_skipped']}")
        print(f"   - Files unsupported: {stats['files_unsupported']}")
        print(f"   - Content types: {stats.get('content_types', {})}")
        
        # Step 4: Verify storage calls
        print(f"\nStorage Operations:")
        print(f"   - store_globule_indexed called {mock_storage.store_globule_indexed.call_count} times")
        
        # Step 5: Examine what was stored
        stored_calls = mock_storage.store_globule_indexed.call_args_list
        
        print(f"\nDetailed Processing Results:")
        for i, call in enumerate(stored_calls):
            globule, original_path = call[0]  # First arg is globule, second is original_file_path
            
            print(f"\n   File {i+1}: {Path(original_path).name}")
            print(f"   - Source: {globule.original_globule.source}")
            print(f"   - Content length: {len(globule.original_globule.raw_text)} chars")
            print(f"   - Has embedding: {len(globule.embedding) > 0}")
            print(f"   - Processing time: {globule.processing_time_ms:.1f}ms")
            print(f"   - File decision: {'Yes' if globule.file_decision else 'No (indexed only)'}")
            print(f"   - Management status: {globule.management_status}")
            
            # Show processor results if available
            provider_metadata = globule.provider_metadata
            if 'processor_type' in provider_metadata:
                print(f"   - Processor: {provider_metadata['processor_type']}")
                print(f"   - Processor confidence: {provider_metadata.get('processor_confidence', 'N/A')}")
            
            # Show parsed data highlights
            if globule.parsed_data:
                print(f"   - Parsed data keys: {list(globule.parsed_data.keys())}")
        
        print(f"\nWalkthrough completed!")
        print("="*60)
        
        # Assertions to verify expected behavior
        assert stats['files_processed'] == 4  # All files were processed
        assert stats['files_indexed'] >= 2  # At least text files should be indexed
        assert mock_storage.store_globule_indexed.call_count >= 2
    
    @pytest.mark.asyncio 
    async def test_processor_router_selection_logic(self, mock_orchestrator_with_processors):
        """Test how the processor router selects processors for different content types."""
        
        print("\n" + "="*60)
        print("PROCESSOR ROUTER SELECTION WALKTHROUGH")
        print("="*60)
        
        router = mock_orchestrator_with_processors.processor_router
        
        # Test different content types
        test_cases = [
            ("document.md", ".md", "Markdown document content"),
            ("notes.txt", ".txt", "Plain text content"),
            ("photo.jpg", ".jpg", "/path/to/photo.jpg"),  # For non-text, content is file path
            ("report.pdf", ".pdf", "/path/to/report.pdf")
        ]
        
        for filename, ext, content in test_cases:
            print(f"\nTesting: {filename}")
            
            # Create mock globule
            from globule.core.models import GlobuleV1
            globule = GlobuleV1(
                raw_text=content,
                source="index",
                initial_context={
                    "filename": filename,
                    "file_extension": ext,
                    "is_file_path": ext not in ['.md', '.txt']
                }
            )
            
            # Test processor confidence scores
            print(f"   Processor confidence scores:")
            for processor in router.get_registered_processors():
                confidence = processor.can_process(globule)
                print(f"   - {processor.get_processor_type()}: {confidence:.2f}")
            
            # Route and process
            try:
                result = await router.route_and_process(globule)
                print(f"   Selected: {result.processor_type} (confidence: {result.confidence:.2f})")
                print(f"   Content type: {result.structured_data.get('content_type', 'unknown')}")
            except Exception as e:
                print(f"   Failed: {e}")
        
        print(f"\nProcessor routing walkthrough completed!")
        print("="*60)


if __name__ == "__main__":
    # Run the walkthrough as a standalone script for debugging
    import asyncio
    
    async def run_walkthrough():
        test = TestCodeWalkthrough()
        
        # Create fixtures manually
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "notes.md").write_text("# Test\nMarkdown content")
            (temp_path / "readme.txt").write_text("Text file content")
            (temp_path / "photo.jpg").write_bytes(b"fake_jpeg")
            (temp_path / "doc.pdf").write_bytes(b"fake_pdf")
            
            # Create mock orchestrator
            router = ProcessorRouter()
            router.register_processor(MockTextProcessor())
            router.register_processor(MockImageProcessor())
            router.register_processor(MockPDFProcessor())
            
            embedding_provider = AsyncMock()
            embedding_provider.embed_single = AsyncMock(return_value=MagicMock(
                embedding=[0.1] * 512,
                processing_time_ms=30.0
            ))
            
            parser_provider = AsyncMock()
            parser_provider.parse = AsyncMock(return_value={"domain": "general"})
            
            storage_manager = AsyncMock()
            
            orchestrator = GlobuleOrchestrator(
                parser_provider=parser_provider,
                embedding_provider=embedding_provider,
                storage_manager=storage_manager,
                processor_router=router
            )
            
            mock_storage = AsyncMock()
            mock_storage.store_globule_indexed = AsyncMock()
            
            # Run tests
            await test.test_complete_indexing_walkthrough(temp_path, orchestrator, mock_storage)
            await test.test_processor_router_selection_logic(orchestrator)
    
    asyncio.run(run_walkthrough())