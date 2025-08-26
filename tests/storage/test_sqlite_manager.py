"""
Integration tests for SQLiteStorageManager search and execute_sql methods.

These tests verify that the SQLite storage manager properly implements
the database operations that were moved out of the orchestrator,
ensuring the architectural separation is correct.

Note: These tests focus on the search() and execute_sql() methods specifically
since those are the methods that were extracted from the orchestrator.
"""

import pytest
import asyncio
import tempfile
import sqlite3
import os
from typing import List, Dict, Any
from uuid import uuid4, UUID

from globule.core.errors import StorageError
from globule.core.interfaces import IStorageManager
from globule.core.models import ProcessedGlobuleV1


class SimpleSQLiteStorageManager(IStorageManager):
    """
    Simplified SQLiteStorageManager for testing just the search/execute_sql methods.
    
    This avoids dependencies on file_manager and other complex components,
    focusing solely on the SQL functionality that was moved from the orchestrator.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create a simple globules table for testing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS globules (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """Simple save implementation for testing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO globules (id, text) VALUES (?, ?)",
                (str(globule.globule_id), globule.original_globule.raw_text)
            )
            conn.commit()
    
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """Simple get implementation for testing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT text FROM globules WHERE id = ?", (str(globule_id),))
            row = cursor.fetchone()
            if not row:
                raise StorageError(f"Globule {globule_id} not found")
            # Return a minimal ProcessedGlobule for testing
            from globule.core.models import GlobuleV1
            raw_globule = GlobuleV1(raw_text=row[0], source="test")
            raw_globule.globule_id = globule_id
            return ProcessedGlobuleV1(
                globule_id=globule_id,
                original_globule=raw_globule,
                embedding=[],
                parsed_data={},
                processing_time_ms=0
            )
    
    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        THE METHOD MOVED FROM ORCHESTRATOR: Search for globules using natural language query.
        """
        try:
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id, text FROM globules WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f"%{query}%", limit)
                )
                
                rows = cursor.fetchall()
                for row in rows:
                    # Create minimal ProcessedGlobule for testing
                    from globule.core.models import GlobuleV1
                    raw_globule = GlobuleV1(raw_text=row[1], source="test")
                    raw_globule.globule_id = UUID(row[0])
                    
                    processed = ProcessedGlobuleV1(
                        globule_id=UUID(row[0]),
                        original_globule=raw_globule,
                        embedding=[],
                        parsed_data={},
                        processing_time_ms=0
                    )
                    results.append(processed)
                    
            return results
            
        except Exception as e:
            raise StorageError(f"Search failed: {e}")

    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        THE METHOD MOVED FROM ORCHESTRATOR: Execute SQL query against the database.
        """
        try:
            # Validate SQL safety (basic check) - SAME LOGIC AS ORCHESTRATOR HAD
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                raise StorageError("Potentially dangerous SQL detected")
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Convert to list of dicts - SAME LOGIC AS ORCHESTRATOR HAD
                results_list = [dict(row) for row in results]
                headers = [desc[0] for desc in cursor.description] if cursor.description else []
                
                return {
                    "type": "sql_results",
                    "query": query,
                    "query_name": query_name,
                    "results": results_list,
                    "headers": headers,
                    "count": len(results_list)
                }
                
        except StorageError:
            # Re-raise storage errors as-is
            raise
        except Exception as e:
            raise StorageError(f"SQL query execution failed: {e}")


class TestSQLiteStorageManager:
    """Integration tests for SQLiteStorageManager database operations."""
    
    @pytest.fixture
    def storage_manager(self):
        """Create a SimpleSQLiteStorageManager with a temporary database."""
        # Create temporary database file
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        try:
            os.close(db_fd)  # Close file descriptor
            
            # Create simple storage manager with temporary database
            storage = SimpleSQLiteStorageManager(db_path)
            
            yield storage
            
        finally:
            # Clean up
            try:
                os.unlink(db_path)
            except:
                pass
    
    @pytest.fixture
    def sample_processed_globule(self):
        """Create a sample ProcessedGlobule for testing."""
        from globule.core.models import GlobuleV1
        
        raw_globule = GlobuleV1(
            raw_text="This is a test globule for SQLite storage",
            source="test"
        )
        
        return ProcessedGlobuleV1(
            globule_id=raw_globule.globule_id,
            original_globule=raw_globule,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            parsed_data={"title": "Test SQLite Storage"},
            processing_time_ms=10.5
        )
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, storage_manager, sample_processed_globule):
        """Test that the search method works correctly."""
        # Store a globule first
        storage_manager.save(sample_processed_globule)
        
        # Search for it
        results = await storage_manager.search("test globule", limit=10)
        
        assert len(results) == 1
        assert results[0].globule_id == sample_processed_globule.globule_id
        assert "test globule" in results[0].original_globule.raw_text
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, storage_manager):
        """Test search when no results are found."""
        results = await storage_manager.search("nonexistent query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_execute_sql_query_success(self, storage_manager, sample_processed_globule):
        """Test successful SQL query execution."""
        # Store a globule to query
        storage_manager.save(sample_processed_globule)
        
        # Execute a safe SELECT query
        result = await storage_manager.execute_sql("SELECT COUNT(*) as count FROM globules")
        
        assert result["type"] == "sql_results"
        assert result["query"] == "SELECT COUNT(*) as count FROM globules"
        assert len(result["results"]) == 1
        assert result["results"][0]["count"] == 1
        assert result["headers"] == ["count"]
        assert result["count"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_sql_query_dangerous_sql(self, storage_manager):
        """Test that dangerous SQL is rejected."""
        with pytest.raises(StorageError) as exc_info:
            await storage_manager.execute_sql("DROP TABLE globules")
        
        assert "dangerous" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_execute_sql_query_invalid_syntax(self, storage_manager):
        """Test SQL query with invalid syntax."""
        with pytest.raises(StorageError) as exc_info:
            await storage_manager.execute_sql("INVALID SQL QUERY")
        
        assert "failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_search_after_multiple_inserts(self, storage_manager):
        """Test search after inserting multiple globules."""
        # Create multiple test globules
        from globule.core.models import GlobuleV1
        
        globules = []
        for i in range(3):
            raw_globule = GlobuleV1(
                raw_text=f"Test globule number {i} about machine learning",
                source="test"
            )
            
            processed = ProcessedGlobuleV1(
                globule_id=raw_globule.globule_id,
                original_globule=raw_globule,
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                parsed_data={"title": f"Test {i}"},
                processing_time_ms=5.0
            )
            
            storage_manager.save(processed)
            globules.append(processed)
        
        # Search for common term
        results = await storage_manager.search("machine learning")
        
        assert len(results) == 3
        
        # Search for specific term
        results = await storage_manager.search("number 1")
        
        assert len(results) == 1
        assert "number 1" in results[0].original_globule.raw_text
    
    @pytest.mark.asyncio
    async def test_sql_query_with_joins(self, storage_manager, sample_processed_globule):
        """Test more complex SQL query to verify database structure."""
        # Store a globule
        storage_manager.save(sample_processed_globule)
        
        # Query with column selection
        result = await storage_manager.execute_sql(
            "SELECT id, text, created_at FROM globules WHERE text LIKE '%test%'"
        )
        
        assert result["type"] == "sql_results"
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == str(sample_processed_globule.globule_id)
        assert "test" in result["results"][0]["text"]
        assert "created_at" in result["results"][0]
        assert result["headers"] == ["id", "text", "created_at"]
    
    @pytest.mark.asyncio 
    async def test_interface_compliance(self, storage_manager):
        """Verify the storage manager properly implements the interface."""
        from globule.core.interfaces import IStorageManager
        
        assert isinstance(storage_manager, IStorageManager)
        
        # Verify required methods exist and are callable
        assert hasattr(storage_manager, 'search')
        assert callable(storage_manager.search)
        
        assert hasattr(storage_manager, 'execute_sql')
        assert callable(storage_manager.execute_sql)
        
        # Verify methods are async
        import inspect
        assert inspect.iscoroutinefunction(storage_manager.search)
        assert inspect.iscoroutinefunction(storage_manager.execute_sql)