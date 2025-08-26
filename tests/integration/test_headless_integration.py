"""
Headless integration tests for the GlobuleOrchestrator.

These tests verify the end-to-end processing of Globules through the
actual orchestration engine without any UI components, proving the
headless architecture works correctly.
"""
import pytest
import asyncio
from uuid import uuid4, UUID
from typing import List, Dict, Any

from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.interfaces import IOrchestrationEngine, IParserProvider, IEmbeddingAdapter, IStorageManager
from globule.core.errors import ParserError, EmbeddingError, StorageError

# Import the actual orchestrator and mock providers
from globule.orchestration.engine import GlobuleOrchestrator
from globule.services.providers_mock import MockParserProvider, MockEmbeddingProvider, MockStorageManager
from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter

# Keep the old NoOp implementations for backward compatibility tests
class NoOpParser(IParserProvider):
    async def parse(self, text: str) -> dict:
        return {"status": "parsed_by_noop", "original_length": len(text)}

class NoOpEmbedder(IEmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [0.0] * 10  # Return a dummy embedding

class InMemoryStorage(IStorageManager):
    _store = {}

    def save(self, globule: ProcessedGlobuleV1) -> None:
        self._store[globule.globule_id] = globule

    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        if globule_id not in self._store:
            raise StorageError(f"Globule {globule_id} not found in in-memory store.")
        return self._store[globule_id]

    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """Simple in-memory search for testing."""
        results = []
        for globule in self._store.values():
            if query.lower() in globule.original_globule.raw_text.lower():
                results.append(globule)
        return results[:limit]
    
    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """Mock SQL execution for testing."""
        return {
            "type": "sql_results",
            "query": query,
            "query_name": query_name, 
            "results": [],
            "headers": [],
            "count": 0
        }

class NoOpOrchestrationEngine(IOrchestrationEngine):
    def __init__(
        self,
        parser: IParserProvider,
        embedder: IEmbeddingAdapter,
        storage: IStorageManager
    ):
        self.parser = parser
        self.embedder = embedder
        self.storage = storage

    async def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
        # Simulate processing steps
        parsed_data = await self.parser.parse(globule.raw_text)
        embedding = await self.embedder.embed(globule.raw_text)

        processed_globule = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=embedding,
            parsed_data=parsed_data,
            processing_time_ms=1.0, # Dummy time
            provider_metadata={
                "parser": "noop",
                "embedder": "noop",
                "storage": "in_memory"
            }
        )
        self.storage.save(processed_globule)
        return processed_globule

@pytest.fixture
def noop_engine():
    """Legacy NoOp engine for backward compatibility."""
    parser = NoOpParser()
    embedder = NoOpEmbedder()
    storage = InMemoryStorage()
    return NoOpOrchestrationEngine(parser, embedder, storage)

@pytest.fixture
def globule_orchestrator():
    """Real GlobuleOrchestrator with mock providers wrapped in real adapters."""
    # 1. Create mock providers
    parser_provider = MockParserProvider()
    embedding_provider = MockEmbeddingProvider()
    storage_manager = MockStorageManager()

    # 2. Wrap mock providers in real adapters
    parsing_adapter = OllamaParsingAdapter(parser_provider)
    embedding_adapter = OllamaEmbeddingAdapter(embedding_provider)

    # 3. Inject adapters into the orchestrator
    return GlobuleOrchestrator(
        parser_provider=parsing_adapter,
        embedding_provider=embedding_adapter,
        storage_manager=storage_manager
    )

async def test_headless_processing_flow(noop_engine: IOrchestrationEngine):
    """Tests the basic end-to-end processing flow of a Globule through the headless engine."""
    raw_globule = GlobuleV1(raw_text="This is a test sentence.", source="cli")
    
    processed_globule = await noop_engine.process(raw_globule)
    
    assert isinstance(processed_globule, ProcessedGlobuleV1)
    assert processed_globule.globule_id == raw_globule.globule_id
    assert processed_globule.original_globule == raw_globule
    assert processed_globule.parsed_data == {"status": "parsed_by_noop", "original_length": len(raw_globule.raw_text)}
    assert processed_globule.embedding == [0.0] * 10
    assert processed_globule.provider_metadata["parser"] == "noop"

    # Verify it was saved
    retrieved_globule = noop_engine.storage.get(raw_globule.globule_id)
    assert retrieved_globule == processed_globule

async def test_globule_orchestrator_headless_processing(globule_orchestrator: GlobuleOrchestrator):
    """
    PHASE 1 CRITICAL TEST: Verify GlobuleOrchestrator works headlessly.
    
    This test proves that our refactored orchestrator can process globules
    end-to-end without any UI components, fulfilling the Phase 1 requirement.
    """
    # Create a raw globule
    raw_globule = GlobuleV1(
        raw_text="This is a headless test of machine learning concepts",
        source="headless_test",
        initial_context={"test_phase": "phase_1", "importance": "critical"}
    )
    
    # Process through the orchestrator (async)
    processed_globule = await globule_orchestrator.process(raw_globule)
    
    # Verify the processing worked
    assert isinstance(processed_globule, ProcessedGlobuleV1)
    assert processed_globule.globule_id == raw_globule.globule_id
    assert processed_globule.original_globule == raw_globule
    
    # Verify mock providers were used via adapters
    assert "OllamaParsingAdapter" in processed_globule.provider_metadata["parser"]
    assert "OllamaEmbeddingAdapter" in processed_globule.provider_metadata["embedder"]
    assert "MockStorageManager" in processed_globule.provider_metadata["storage"]
    
    # Verify parsed data structure (from MockParserProvider)
    assert "title" in processed_globule.parsed_data
    assert "category" in processed_globule.parsed_data
    assert "domain" in processed_globule.parsed_data
    assert processed_globule.parsed_data["metadata"]["mock"] is True
    
    # Verify embedding was generated
    assert len(processed_globule.embedding) > 0
    assert all(isinstance(x, float) for x in processed_globule.embedding)
    
    # Verify processing time was recorded
    assert processed_globule.processing_time_ms >= 0

@pytest.mark.asyncio
async def test_orchestrator_async_business_logic(globule_orchestrator: GlobuleOrchestrator):
    """
    Test the async business logic methods in the orchestrator.
    
    This validates that the orchestrator properly handles async workflows
    that were previously embedded in the TUI.
    """
    # Test thought capture
    captured = await globule_orchestrator.capture_thought(
        raw_text="Async test thought about distributed systems",
        source="headless_test",
        context={"async": True}
    )
    
    assert isinstance(captured, ProcessedGlobuleV1)
    assert captured.original_globule.raw_text == "Async test thought about distributed systems"
    assert captured.original_globule.initial_context["async"] is True
    
    # Test search functionality
    search_results = await globule_orchestrator.search_globules("distributed systems")
    assert isinstance(search_results, list)
    
    # Test globule retrieval
    retrieved = await globule_orchestrator.get_globule(captured.globule_id)
    assert retrieved is not None
    assert retrieved.globule_id == captured.globule_id
    
    # Test query execution
    query_result = await globule_orchestrator.execute_query("find thoughts about systems")
    assert query_result["type"] == "search_results"
    assert "results" in query_result
    assert "count" in query_result

async def test_orchestrator_implements_interface(globule_orchestrator: GlobuleOrchestrator):
    """Verify orchestrator properly implements the IOrchestrationEngine interface."""
    assert isinstance(globule_orchestrator, IOrchestrationEngine)
    
    # Test interface method exists and is callable
    assert hasattr(globule_orchestrator, 'process')
    assert callable(globule_orchestrator.process)
    
    # Test that process method is async
    import inspect
    assert inspect.iscoroutinefunction(globule_orchestrator.process)
    
    # Test it can process a globule through the interface
    test_globule = GlobuleV1(raw_text="Interface test", source="test")
    result = await globule_orchestrator.process(test_globule)
    assert isinstance(result, ProcessedGlobuleV1)
