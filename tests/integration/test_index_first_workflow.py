"""
Integration tests for the Index-First architectural workflow.

Tests both Phase A (indexing) and Phase B (organization) of the new architecture.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from globule.core.api import GlobuleAPI
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.orchestration.engine import GlobuleOrchestrator


class TestIndexFirstWorkflow:
    """Test the complete Index-First workflow."""
    
    @pytest.fixture
    async def temp_content_dir(self):
        """Create a temporary directory with test content files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files with different types
            (temp_path / "note1.md").write_text("""
# Machine Learning Concepts
This document covers basic ML concepts including supervised learning, neural networks, and deep learning architectures.
""")
            
            (temp_path / "note2.txt").write_text("""
Python programming tips:
- Use list comprehensions for efficient iteration
- Leverage async/await for concurrent operations
- Apply type hints for better code documentation
""")
            
            (temp_path / "project.py").write_text("""
# Simple Python script
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

if __name__ == "__main__":
    print(calculate_fibonacci(10))
""")
            
            # Create a subdirectory with more content
            subdir = temp_path / "papers"
            subdir.mkdir()
            
            (subdir / "research.md").write_text("""
# Research Paper Notes
This contains notes on recent research in artificial intelligence, focusing on transformer architectures and attention mechanisms.
""")
            
            yield temp_path
    
    @pytest.fixture
    async def mock_storage_manager(self):
        """Create a mock storage manager for testing."""
        storage = AsyncMock(spec=SQLiteStorageManager)
        storage.initialize = AsyncMock()
        storage.store_globule_indexed = AsyncMock(return_value="test-id")
        storage.get_unmanaged_globules = AsyncMock(return_value=[])
        return storage
    
    @pytest.fixture
    async def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        orchestrator = AsyncMock(spec=GlobuleOrchestrator)
        
        # Mock the process_globule_for_indexing method
        async def mock_process(enriched_input):
            from globule.core.models import ProcessedGlobuleV1, GlobuleV1
            from uuid import uuid4
            from datetime import datetime
            
            globule = GlobuleV1(
                globule_id=uuid4(),
                raw_text=enriched_input.original_text,
                source=enriched_input.source
            )
            
            return ProcessedGlobuleV1(
                globule_id=globule.globule_id,
                original_globule=globule,
                embedding=[0.1] * 512,  # Mock embedding
                parsed_data={"domain": "test", "category": "note"},
                file_decision=None,  # No file decision for indexed globules
                processing_time_ms=100.0,
                provider_metadata={
                    'embedding_confidence': 0.9,
                    'parsing_confidence': 0.8,
                    'orchestration_strategy': 'index_parallel'
                }
            )
        
        orchestrator.process_globule_for_indexing = AsyncMock(side_effect=mock_process)
        return orchestrator
    
    @pytest.fixture
    async def test_api(self, mock_storage_manager, mock_orchestrator):
        """Create a GlobuleAPI instance with mocked dependencies."""
        api = GlobuleAPI(storage=mock_storage_manager, orchestrator=mock_orchestrator)
        return api
    
    @pytest.mark.asyncio
    async def test_index_path_basic_functionality(self, test_api, temp_content_dir, mock_storage_manager):
        """Test basic indexing functionality."""
        # Execute indexing
        stats = await test_api.index_path(str(temp_content_dir))
        
        # Verify statistics
        assert stats["files_processed"] == 4  # 4 files total
        assert stats["files_indexed"] == 4    # All should be indexed
        assert stats["files_skipped"] == 0
        assert stats["files_failed"] == 0
        assert stats["total_size_bytes"] > 0
        
        # Verify storage was called for each file
        assert mock_storage_manager.store_globule_indexed.call_count == 4
        
        # Verify the paths passed to storage
        calls = mock_storage_manager.store_globule_indexed.call_args_list
        indexed_paths = [call[0][1] for call in calls]  # Second argument is original_file_path
        
        # Should include all our test files
        assert any("note1.md" in path for path in indexed_paths)
        assert any("note2.txt" in path for path in indexed_paths)
        assert any("project.py" in path for path in indexed_paths)
        assert any("research.md" in path for path in indexed_paths)
    
    @pytest.mark.asyncio
    async def test_index_path_file_type_filtering(self, test_api, temp_content_dir):
        """Test file type filtering during indexing."""
        # Index only markdown files
        stats = await test_api.index_path(str(temp_content_dir), file_types=['.md'])
        
        # Should only process markdown files
        assert stats["files_processed"] == 2  # note1.md and research.md
        assert stats["files_indexed"] == 2
        assert stats["files_skipped"] == 2   # note2.txt and project.py skipped
    
    @pytest.mark.asyncio
    async def test_index_path_error_handling(self, test_api):
        """Test error handling for invalid paths."""
        # Test non-existent path
        with pytest.raises(ValueError, match="Path does not exist"):
            await test_api.index_path("/non/existent/path")
        
        # Test file instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                await test_api.index_path(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_organize_repository_no_unmanaged_content(self, test_api, mock_storage_manager):
        """Test organize when no unmanaged content exists."""
        # Mock no unmanaged globules
        mock_storage_manager.get_unmanaged_globules.return_value = []
        
        result = await test_api.organize_repository(dry_run=True)
        
        assert result["status"] == "no_unmanaged_globules"
        assert result["count"] == 0
    
    @pytest.mark.asyncio
    async def test_organize_repository_with_content(self, test_api, mock_storage_manager):
        """Test organize with unmanaged content."""
        # Mock unmanaged globules
        from globule.core.models import ProcessedGlobuleV1, GlobuleV1
        from uuid import uuid4
        
        mock_globules = []
        for i in range(5):
            globule = GlobuleV1(
                globule_id=uuid4(),
                raw_text=f"Test content {i} about machine learning and neural networks",
                source="index"
            )
            
            processed = ProcessedGlobuleV1(
                globule_id=globule.globule_id,
                original_globule=globule,
                embedding=[0.1] * 512,
                parsed_data={"domain": "ml", "category": "notes"},
                file_decision=None,  # Unmanaged
                processing_time_ms=100.0
            )
            mock_globules.append(processed)
        
        mock_storage_manager.get_unmanaged_globules.return_value = mock_globules
        
        # Mock clustering analysis
        with pytest.mock.patch('globule.services.clustering.semantic_clustering.SemanticClusteringEngine') as mock_engine:
            mock_cluster = MagicMock()
            mock_cluster.label = "Machine Learning Notes"
            mock_cluster.description = "Notes about ML and AI"
            mock_cluster.keywords = ["machine", "learning", "neural", "networks"]
            mock_cluster.size = 5
            mock_cluster.confidence_score = 0.8
            
            mock_analysis = MagicMock()
            mock_analysis.clusters = [mock_cluster]
            
            mock_engine_instance = AsyncMock()
            mock_engine_instance.analyze_semantic_clusters.return_value = mock_analysis
            mock_engine.return_value = mock_engine_instance
            
            result = await test_api.organize_repository(dry_run=True)
        
        # Verify results
        assert result["unmanaged_count"] == 5
        assert result["clusters_found"] == 1
        assert result["dry_run"] is True
        
        structure = result["proposed_structure"]
        assert structure["type"] == "clustered"
        assert "machine-learning-notes" in structure["directories"]
        
        cluster_info = structure["directories"]["machine-learning-notes"]
        assert cluster_info["globule_count"] == 5
        assert cluster_info["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_directory_name_sanitization(self, test_api):
        """Test directory name sanitization."""
        # Test various problematic names
        assert test_api._sanitize_directory_name("Machine Learning & AI!") == "machine-learning-ai"
        assert test_api._sanitize_directory_name("Notes (2024)") == "notes-2024"
        assert test_api._sanitize_directory_name("  Spaced   Out  ") == "spaced-out"
        assert test_api._sanitize_directory_name("Very-Long-Name-That-Should-Be-Truncated-At-Some-Point")[:50]
    
    def test_processed_globule_management_status_properties(self):
        """Test the new management status properties."""
        from globule.core.models import ProcessedGlobuleV1, GlobuleV1, FileDecisionV1
        from uuid import uuid4
        
        # Test indexed globule (no file_decision)
        globule = GlobuleV1(globule_id=uuid4(), raw_text="test", source="index")
        
        indexed_processed = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=[0.1] * 512,
            parsed_data={},
            file_decision=None,
            processing_time_ms=100.0,
            provider_metadata={'original_file_path': '/path/to/file.md'}
        )
        
        assert indexed_processed.is_indexed_only is True
        assert indexed_processed.is_managed is False
        assert indexed_processed.management_status == "indexed"
        assert indexed_processed.original_file_path == "/path/to/file.md"
        
        # Test managed globule (has file_decision)
        managed_processed = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=[0.1] * 512,
            parsed_data={},
            file_decision=FileDecisionV1(
                semantic_path="notes",
                filename="test.md",
                confidence=0.8
            ),
            processing_time_ms=100.0,
            provider_metadata={}
        )
        
        assert managed_processed.is_indexed_only is False
        assert managed_processed.is_managed is True
        assert managed_processed.management_status == "managed"
        
        # Test captured globule (no file_decision, no original_file_path)
        captured_processed = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=[0.1] * 512,
            parsed_data={},
            file_decision=None,
            processing_time_ms=100.0,
            provider_metadata={}
        )
        
        assert captured_processed.management_status == "captured"


@pytest.mark.integration
class TestIndexFirstWorkflowIntegration:
    """Full integration tests with real components (slower)."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_real_storage(self):
        """Test the complete workflow with real SQLite storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test content
            test_file = temp_path / "test.md"
            test_file.write_text("# Test Document\nThis is a test document for indexing.")
            
            # Create storage with in-memory database
            storage = SQLiteStorageManager(db_path=Path(':memory:'))
            await storage.initialize()
            
            # Create mock orchestrator
            orchestrator = AsyncMock()
            
            async def mock_process(enriched_input):
                from globule.core.models import ProcessedGlobuleV1, GlobuleV1
                from uuid import uuid4
                
                globule = GlobuleV1(
                    globule_id=uuid4(),
                    raw_text=enriched_input.original_text,
                    source=enriched_input.source
                )
                
                return ProcessedGlobuleV1(
                    globule_id=globule.globule_id,
                    original_globule=globule,
                    embedding=[0.1] * 512,
                    parsed_data={"domain": "test"},
                    file_decision=None,
                    processing_time_ms=100.0
                )
            
            orchestrator.process_globule_for_indexing = AsyncMock(side_effect=mock_process)
            
            # Create API and test indexing
            api = GlobuleAPI(storage=storage, orchestrator=orchestrator)
            stats = await api.index_path(str(temp_path))
            
            # Verify results
            assert stats["files_indexed"] == 1
            
            # Verify storage contains the indexed globule
            unmanaged = await storage.get_unmanaged_globules()
            assert len(unmanaged) == 1
            assert unmanaged[0].original_globule.raw_text == "# Test Document\nThis is a test document for indexing."
            
            await storage.close()