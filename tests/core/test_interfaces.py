"""
Contract compliance tests for the Abstract Base Classes in core.interfaces.

These tests ensure that the interfaces are well-defined and can be implemented.
"""
import pytest
from uuid import uuid4
from typing import List, Dict, Any

from globule.core.interfaces import (
    IParserProvider,
    IEmbeddingAdapter,
    IStorageManager,
    IOrchestrationEngine,
    ISchemaManager
)
from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.errors import ParserError, EmbeddingError, StorageError

# Dummy implementations for testing contract compliance

class DummyParser(IParserProvider):
    async def parse(self, text: str) -> dict:
        if not text:
            raise ParserError("Input text cannot be empty")
        return {"parsed": True, "text_length": len(text)}

class DummyEmbedder(IEmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        if not text:
            raise EmbeddingError("Input text cannot be empty")
        return [len(text) / 100.0]

class DummyStorage(IStorageManager):
    _storage = {}

    def save(self, globule: ProcessedGlobuleV1) -> None:
        if not globule.globule_id:
            raise StorageError("Globule ID is required")
        self._storage[globule.globule_id] = globule

    def get(self, globule_id: str) -> ProcessedGlobuleV1:
        if globule_id not in self._storage:
            raise StorageError(f"Globule {globule_id} not found")
        return self._storage[globule_id]

    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        return list(self._storage.values())[:limit]

    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        return {"results": []}

class DummySchemaManager(ISchemaManager):
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        return {"name": schema_name}

    def detect_schema_for_text(self, text: str) -> str | None:
        return "default"

    def get_available_schemas(self) -> List[str]:
        return ["default", "test"]

class DummyOrchestrator(IOrchestrationEngine):
    def __init__(self, parser, embedder, storage):
        self.parser = parser
        self.embedder = embedder
        self.storage = storage

    async def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
        parsed_data = await self.parser.parse(globule.raw_text)
        embedding = await self.embedder.embed(globule.raw_text)
        
        processed_globule = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=embedding,
            parsed_data=parsed_data,
            processing_time_ms=50.0
        )
        self.storage.save(processed_globule)
        return processed_globule

# Tests

@pytest.mark.asyncio
async def test_dummy_parser_compliance():
    """Tests that DummyParser correctly implements IParserProvider."""
    parser: IParserProvider = DummyParser()
    result = await parser.parse("test")
    assert result == {"parsed": True, "text_length": 4}
    with pytest.raises(ParserError):
        await parser.parse("")

@pytest.mark.asyncio
async def test_dummy_embedder_compliance():
    """Tests that DummyEmbedder correctly implements IEmbeddingAdapter."""
    embedder: IEmbeddingAdapter = DummyEmbedder()
    result = await embedder.embed("test")
    assert result == [0.04]
    with pytest.raises(EmbeddingError):
        await embedder.embed("")

@pytest.mark.asyncio
async def test_dummy_storage_compliance():
    """Tests that DummyStorage correctly implements IStorageManager."""
    storage: IStorageManager = DummyStorage()
    raw_globule = GlobuleV1(raw_text="test", source="test")
    processed_globule = ProcessedGlobuleV1(
        globule_id=raw_globule.globule_id,
        original_globule=raw_globule,
        embedding=[0.1],
        parsed_data={},
        processing_time_ms=10.0
    )
    
    storage.save(processed_globule)
    retrieved = storage.get(processed_globule.globule_id)
    assert retrieved == processed_globule

    with pytest.raises(StorageError):
        storage.get(uuid4())

@pytest.mark.asyncio
async def test_dummy_orchestrator_compliance():
    """Tests that the dummy components work together via their interfaces."""
    storage = DummyStorage()
    orchestrator: IOrchestrationEngine = DummyOrchestrator(DummyParser(), DummyEmbedder(), storage)
    raw_globule = GlobuleV1(raw_text="orchestration test", source="test")
    
    result = await orchestrator.process(raw_globule)
    
    assert isinstance(result, ProcessedGlobuleV1)
    assert result.original_globule == raw_globule
    assert result.parsed_data == {"parsed": True, "text_length": 18}
    assert result.embedding == [0.18]
    # Verify it was saved
    assert storage.get(raw_globule.globule_id) is not None
