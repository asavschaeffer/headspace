"""
Unit tests for SqliteStorageAdapter.

These tests verify the adapter correctly implements the IStorageManager interface
using SQLite as the backend. Uses in-memory databases for isolation and speed.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from globule.storage.sqlite_adapter import SqliteStorageAdapter
from globule.core.models import ProcessedGlobuleV1, GlobuleV1, FileDecisionV1
from globule.core.errors import StorageError
from globule.core.interfaces import IStorageManager


class TestSqliteStorageAdapter:
    """Test suite for SqliteStorageAdapter"""
    
    @pytest.fixture
    async def adapter(self):
        """Create a SqliteStorageAdapter with in-memory database for testing."""
        # Use in-memory database for tests
        adapter = SqliteStorageAdapter(db_path=Path(":memory:"))
        await adapter.initialize()
        yield adapter
        await adapter.close()
    
    @pytest.fixture
    def valid_processed_globule(self):
        """Create a single, perfectly valid ProcessedGlobuleV1 for testing."""
        # Create the original globule first
        original_globule = GlobuleV1(
            raw_text="This is a test globule for storage testing",
            source="test",
            initial_context={"test": True}
        )
        
        # Create the processed globule with all required fields
        return ProcessedGlobuleV1(
            globule_id=original_globule.globule_id,
            original_globule=original_globule,
            embedding=[0.1, 0.2, 0.3, 0.4],
            parsed_data={"title": "Test", "category": "unit_test", "domain": "testing"},
            file_decision=FileDecisionV1(
                semantic_path="testing/unit_test",
                filename="test_globule.md",
                confidence=0.85
            ),
            processing_time_ms=150.0,
            provider_metadata={"test": "adapter"}
        )
    
    def test_implements_interface(self, adapter):
        """Test that SqliteStorageAdapter implements IStorageManager interface."""
        assert isinstance(adapter, IStorageManager)
        
        # Check all required methods exist
        assert hasattr(adapter, 'save')
        assert hasattr(adapter, 'get')
        assert hasattr(adapter, 'search')
        assert hasattr(adapter, 'execute_sql')
        
        # Check async methods are coroutines
        import inspect
        assert inspect.iscoroutinefunction(adapter.search)
        assert inspect.iscoroutinefunction(adapter.execute_sql)
    
    def test_adapter_initialization(self):
        """Test adapter initialization with different configurations."""
        # Test with default path
        adapter1 = SqliteStorageAdapter()
        assert adapter1.db_path is not None
        
        # Test with custom path
        custom_path = Path("/tmp/test.db")
        adapter2 = SqliteStorageAdapter(db_path=custom_path)
        assert adapter2.db_path == custom_path
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_globule(self, adapter, valid_processed_globule):
        """Test storing and retrieving a globule."""
        # Mock file manager to avoid actual file operations
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            # Store the globule
            globule_id = await adapter.store_globule(valid_processed_globule)
            assert globule_id == str(valid_processed_globule.globule_id)
            
            # Verify file manager was called
            mock_determine.assert_called_once()
            mock_save_temp.assert_called_once()
            mock_commit.assert_called_once()
            
            # Retrieve the globule
            retrieved = await adapter.get_globule(globule_id)
            assert retrieved is not None
            assert str(retrieved.globule_id) == str(valid_processed_globule.globule_id)
            assert retrieved.original_globule.raw_text == valid_processed_globule.original_globule.raw_text
            # Use approximate equality for floating point comparisons
            assert len(retrieved.embedding) == len(valid_processed_globule.embedding)
            for i, (a, b) in enumerate(zip(retrieved.embedding, valid_processed_globule.embedding)):
                assert abs(a - b) < 1e-6, f"Embedding mismatch at index {i}: {a} != {b}"
            assert retrieved.parsed_data == valid_processed_globule.parsed_data
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, adapter, valid_processed_globule):
        """Test search functionality."""
        # Store a globule first
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            await adapter.store_globule(valid_processed_globule)
        
        # Search for the globule
        results = await adapter.search("test globule", limit=10)
        assert len(results) >= 1
        assert any(str(result.globule_id) == str(valid_processed_globule.globule_id) for result in results)
        
        # Search with no results
        no_results = await adapter.search("nonexistent query", limit=10)
        assert len(no_results) == 0
    
    @pytest.mark.asyncio
    async def test_execute_sql_valid_query(self, adapter, valid_processed_globule):
        """Test executing valid SQL queries."""
        # Store a globule first
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            await adapter.store_globule(valid_processed_globule)
        
        # Execute valid SELECT query
        result = await adapter.execute_sql("SELECT COUNT(*) as count FROM globules", "Count Test")
        assert result["type"] == "sql_results"
        assert result["query_name"] == "Count Test"
        assert "results" in result
        assert "headers" in result
        assert result["count"] >= 1
    
    @pytest.mark.asyncio
    async def test_execute_sql_dangerous_query(self, adapter):
        """Test that dangerous SQL queries are rejected."""
        dangerous_queries = [
            "DROP TABLE globules",
            "DELETE FROM globules",
            "UPDATE globules SET text = 'hacked'",
            "INSERT INTO globules VALUES (...)",
            "TRUNCATE TABLE globules",
            "ALTER TABLE globules ADD COLUMN evil TEXT"
        ]
        
        for dangerous_query in dangerous_queries:
            with pytest.raises(StorageError, match="Potentially dangerous SQL detected"):
                await adapter.execute_sql(dangerous_query)
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_globule(self, adapter):
        """Test retrieving a non-existent globule."""
        fake_id = str(uuid4())
        result = await adapter.get_globule(fake_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_sync_save_method(self, adapter, valid_processed_globule):
        """Test the synchronous save method by using async methods directly."""
        # Mock file manager
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            # Use async methods directly in async test
            await adapter.store_globule(valid_processed_globule)
            
            # Verify it was stored
            retrieved = await adapter.get_globule(str(valid_processed_globule.globule_id))
            assert retrieved is not None
            assert str(retrieved.globule_id) == str(valid_processed_globule.globule_id)
    
    @pytest.mark.asyncio
    async def test_sync_get_method(self, adapter, valid_processed_globule):
        """Test the synchronous get method by using async methods directly."""
        # Store a globule first
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            await adapter.store_globule(valid_processed_globule)
        
        # Retrieve using async method directly in async test
        retrieved = await adapter.get_globule(str(valid_processed_globule.globule_id))
        assert retrieved is not None
        assert str(retrieved.globule_id) == str(valid_processed_globule.globule_id)
    
    def test_sync_get_nonexistent(self, adapter):
        """Test sync get method with non-existent globule."""
        fake_id = uuid4()
        with pytest.raises(StorageError, match="not found"):
            adapter.get(fake_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self):
        """Test error handling when database operations fail."""
        # Create adapter with invalid path to trigger errors
        adapter = SqliteStorageAdapter(db_path=Path("/invalid/path/database.db"))
        
        with pytest.raises(StorageError, match="Failed to initialize SQLite storage"):
            await adapter.initialize()
    
    @pytest.mark.asyncio
    async def test_connection_management(self, adapter):
        """Test connection management."""
        # Connection exists after initialization
        assert adapter._connection is not None
        
        # Getting connection returns existing connection
        conn = await adapter._get_connection()
        assert conn is not None
        assert adapter._connection is conn
        
        # Getting connection again returns same instance
        conn2 = await adapter._get_connection()
        assert conn2 is conn
        
        # Closing clears connection
        await adapter.close()
        assert adapter._connection is None
    
    @pytest.mark.asyncio
    async def test_vector_normalization(self, adapter):
        """Test vector normalization functionality."""
        # Test normal vector
        vector = np.array([3.0, 4.0], dtype=np.float32)
        normalized = adapter._normalize_vector(vector)
        expected_norm = np.linalg.norm(normalized)
        assert abs(expected_norm - 1.0) < 1e-6
        
        # Test zero vector
        zero_vector = np.array([0.0, 0.0], dtype=np.float32)
        normalized_zero = adapter._normalize_vector(zero_vector)
        np.testing.assert_array_equal(normalized_zero, zero_vector)
        
        # Test None input
        assert adapter._normalize_vector(None) is None
    
    @pytest.mark.asyncio
    async def test_row_to_globule_conversion(self, adapter, valid_processed_globule):
        """Test conversion from database row to ProcessedGlobule."""
        # Store and retrieve to test conversion
        with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
             patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
             patch.object(adapter._file_manager, 'commit_file') as mock_commit:
            
            mock_determine.return_value = Path("/tmp/globule_storage/test_path.md")
            mock_save_temp.return_value = Path("/tmp/temp_file.md")
            
            await adapter.store_globule(valid_processed_globule)
            retrieved = await adapter.get_globule(str(valid_processed_globule.globule_id))
            
            # Verify all fields are correctly converted
            assert str(retrieved.globule_id) == str(valid_processed_globule.globule_id)
            assert retrieved.original_globule.raw_text == valid_processed_globule.original_globule.raw_text
            assert retrieved.embedding == pytest.approx(valid_processed_globule.embedding)
            assert retrieved.parsed_data == valid_processed_globule.parsed_data
            assert retrieved.processing_time_ms == valid_processed_globule.processing_time_ms
    
    @pytest.mark.asyncio
    async def test_storage_error_translation(self, adapter):
        """Test that various exceptions are properly translated to StorageError."""
        # Test SQL execution error
        with pytest.raises(StorageError):
            await adapter.execute_sql("SELECT * FROM nonexistent_table")
        
        # Test search error with corrupted state
        adapter._connection = None
        with patch.object(adapter, '_get_connection', side_effect=Exception("DB Connection failed")):
            with pytest.raises(StorageError, match="Search failed"):
                await adapter.search("test query")


class TestSqliteStorageAdapterIntegration:
    """Integration tests for SqliteStorageAdapter with real file system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for integration tests."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)  # Close the file descriptor, we just need the path
        yield Path(path)
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_real_db(self, temp_db_path):
        """Test full lifecycle with a real database file."""
        adapter = SqliteStorageAdapter(db_path=temp_db_path)
        
        try:
            await adapter.initialize()
            
            # Create test globule using the same fixture pattern
            original_globule = GlobuleV1(
                raw_text="Integration test globule",
                source="integration_test"
            )
            
            test_globule = ProcessedGlobuleV1(
                globule_id=original_globule.globule_id,
                original_globule=original_globule,
                embedding=[0.5, 0.6, 0.7],
                parsed_data={"type": "integration_test"},
                processing_time_ms=125.0
            )
            
            # Mock file operations for integration test
            with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
                 patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
                 patch.object(adapter._file_manager, 'commit_file') as mock_commit:
                
                mock_determine.return_value = Path("/tmp/globule_storage/integration_test.md")
                mock_save_temp.return_value = Path("/tmp/temp_integration.md")
                
                # Store globule
                globule_id = await adapter.store_globule(test_globule)
                assert globule_id == str(test_globule.globule_id)
                
                # Retrieve globule
                retrieved = await adapter.get_globule(globule_id)
                assert retrieved is not None
                assert retrieved.original_globule.raw_text == test_globule.original_globule.raw_text
                
                # Search for globule
                search_results = await adapter.search("integration test")
                assert len(search_results) >= 1
                assert any(str(g.globule_id) == globule_id for g in search_results)
                
                # Execute SQL query
                sql_result = await adapter.execute_sql("SELECT COUNT(*) as total FROM globules")
                assert sql_result["type"] == "sql_results"
                assert len(sql_result["results"]) == 1
        
        finally:
            await adapter.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_db_path):
        """Test concurrent database operations."""
        adapter = SqliteStorageAdapter(db_path=temp_db_path)
        
        try:
            await adapter.initialize()
            
            # Create multiple test globules using the correct model structure
            globules = []
            for i in range(5):
                original = GlobuleV1(raw_text=f"Concurrent test {i}", source="concurrent_test")
                globule = ProcessedGlobuleV1(
                    globule_id=original.globule_id,
                    original_globule=original,
                    embedding=[0.1 * i, 0.2 * i],
                    parsed_data={"index": i},
                    processing_time_ms=125.0
                )
                globules.append(globule)
            
            # Mock file operations
            with patch.object(adapter._file_manager, 'determine_path') as mock_determine, \
                 patch.object(adapter._file_manager, 'save_to_temp') as mock_save_temp, \
                 patch.object(adapter._file_manager, 'commit_file') as mock_commit:
                
                mock_determine.return_value = Path("/tmp/globule_storage/concurrent_test.md")
                mock_save_temp.return_value = Path("/tmp/temp_concurrent.md")
                
                # Store all globules concurrently
                store_tasks = [adapter.store_globule(g) for g in globules]
                stored_ids = await asyncio.gather(*store_tasks)
                
                assert len(stored_ids) == 5
                assert all(id for id in stored_ids)
                
                # Retrieve all globules concurrently
                retrieve_tasks = [adapter.get_globule(id) for id in stored_ids]
                retrieved_globules = await asyncio.gather(*retrieve_tasks)
                
                assert len(retrieved_globules) == 5
                assert all(g is not None for g in retrieved_globules)
        
        finally:
            await adapter.close()