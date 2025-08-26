"""
Mock providers implementing the core interfaces for testing and Phase 1.

These providers implement the IParserProvider, IEmbeddingAdapter, and IStorageManager
interfaces with dummy implementations for use during Phase 1 testing.
"""

import asyncio
import time
from typing import Dict, Any, List
from uuid import UUID

from globule.core.interfaces import IParserProvider, IStorageManager
from globule.core.models import ProcessedGlobuleV1, StructuredQuery
from globule.core.errors import ParserError, StorageError

# Import the proper mock embedding adapter
from globule.services.embedding.mock_adapter import MockEmbeddingAdapter
# Provide backward compatibility alias
MockEmbeddingProvider = MockEmbeddingAdapter


class MockParserProvider(IParserProvider):
    """Mock parser provider for Phase 1 testing."""
    
    async def parse(self, text: str) -> dict:
        """Return mock parsed data."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        return {
            "title": text[:50] + "..." if len(text) > 50 else text,
            "category": "note", 
            "domain": "general",
            "keywords": [],
            "entities": [],
            "metadata": {
                "mock": True,
                "parser_version": "mock-1.0.0"
            }
        }


# MockEmbeddingProvider is now imported from mock_adapter.py
# This maintains backward compatibility while using the proper BaseEmbeddingAdapter implementation


class MockStorageManager(IStorageManager):
    """Mock storage manager for Phase 1 testing."""
    
    def __init__(self):
        self._storage = {}  # In-memory storage for testing
    
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """Save globule to in-memory storage."""
        self._storage[globule.globule_id] = globule
    
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """Retrieve globule from in-memory storage."""
        if globule_id not in self._storage:
            raise StorageError(f"Globule {globule_id} not found")
        return self._storage[globule_id]
    
    def list_all(self) -> List[ProcessedGlobuleV1]:
        """Return all stored globules."""
        return list(self._storage.values())
    
    def clear(self) -> None:
        """Clear all stored globules (for testing)."""
        self._storage.clear()
    
    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """Mock search implementation."""
        # Simple mock search - return globules containing query text
        results = []
        for globule in self._storage.values():
            if query.lower() in globule.original_globule.raw_text.lower():
                results.append(globule)
        
        return results[:limit]
    
    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """Mock SQL execution."""
        return {
            "type": "sql_results", 
            "query": query,
            "query_name": query_name,
            "results": [{"mock": "result", "count": len(self._storage)}],
            "headers": ["mock", "count"],
            "count": 1
        }
    
    async def query_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]:
        """Mock structured query implementation."""
        # Simple mock implementation - filter by domain and category
        results = []
        for globule in self._storage.values():
            parsed_data = globule.parsed_data
            
            # Filter by domain if specified
            if query.domain and parsed_data.get("domain") != query.domain:
                continue
            
            # Filter by category if specified in filters
            if query.filters and "category" in query.filters:
                if parsed_data.get("category") != query.filters["category"]:
                    continue
            
            # Filter by processor type if specified in filters
            if query.filters and "processor_type" in query.filters:
                provider_metadata = globule.provider_metadata or {}
                if provider_metadata.get("processor_type") != query.filters["processor_type"]:
                    continue
            
            results.append(globule)
            
            # Apply limit
            if len(results) >= query.limit:
                break
        
        return results