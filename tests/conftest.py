"""
Test configuration and fixtures for Globule tests.

Provides real database fixtures and test data for proper integration testing.
All integration tests use persistent databases and real data.
"""

import pytest
import numpy as np
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from globule.core.models import ProcessedGlobuleV1, FileDecisionV1, GlobuleV1
# from globule.core.models import ProcessedGlobule, FileDecision # TODO: Migrate fixtures to V1 contracts
from globule.storage.sqlite_manager import SQLiteStorageManager
from tests.utils.embedding_generator import (
    generate_all_test_embeddings, 
    generate_performance_embeddings,
    load_test_embeddings
)


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager for unit tests that don't require real database."""
    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.store_globule = AsyncMock(return_value="test-id-123")
    storage.get_globule = AsyncMock()
    storage.get_recent_globules = AsyncMock(return_value=[])
    storage.search_by_embedding = AsyncMock(return_value=[])
    storage.update_globule = AsyncMock(return_value=True)
    storage.delete_globule = AsyncMock(return_value=True)
    return storage


@pytest.fixture
async def real_storage_manager(tmp_path):
    """Create a real SQLite storage manager with temporary database for integration tests."""
    db_path = tmp_path / "integration_test.db"
    storage = SQLiteStorageManager(db_path)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def real_test_data():
    """Load real test data with pre-computed deterministic embeddings."""
    config = load_test_embeddings()
    embeddings = generate_all_test_embeddings()
    
    globules = []
    for globule_config in config["test_globules"]:
        globule_data = {
            "id": globule_config["id"],
            "text": globule_config["text"],
            "embedding": embeddings[globule_config["id"]].tolist(),
            "embedding_confidence": globule_config["embedding_confidence"],
            "parsed_data": globule_config["parsed_data"],
            "parsing_confidence": globule_config["parsing_confidence"],
            "file_decision": {
                "semantic_path": globule_config["file_decision"]["semantic_path"],
                "filename": globule_config["file_decision"]["filename"],
                "metadata": {},
                "confidence": globule_config["file_decision"]["confidence"],
                "alternative_paths": []
            },
            "created_at": (datetime.now() - timedelta(days=globule_config["days_ago"])).isoformat()
        }
        globules.append(globule_data)
    
    return {"globules": globules}


@pytest.fixture
def large_test_dataset():
    """Generate a large dataset for performance testing (100k+ records)."""
    # Generate 100k records with deterministic embeddings for proper performance testing
    dataset = generate_performance_embeddings(count=100000)
    return {"globules": dataset}


# @pytest.fixture
# async def populated_real_storage(real_storage_manager, real_test_data):
#     """Create a real storage manager populated with realistic test data."""
#     for globule_data in real_test_data["globules"]:
#         # Convert dict to ProcessedGlobule with relative paths
#         globule = ProcessedGlobule(
#             id=globule_data["id"],
#             text=globule_data["text"],
#             embedding=np.array(globule_data["embedding"], dtype=np.float32),
#             embedding_confidence=globule_data["embedding_confidence"],
#             parsed_data=globule_data["parsed_data"],
#             parsing_confidence=globule_data["parsing_confidence"],
#             file_decision=FileDecision(
#                 semantic_path=Path(globule_data["file_decision"]["semantic_path"]),
#                 filename=globule_data["file_decision"]["filename"],
#                 metadata=globule_data["file_decision"]["metadata"],
#                 confidence=globule_data["file_decision"]["confidence"],
#                 alternative_paths=globule_data["file_decision"]["alternative_paths"]
#             ),
#             orchestration_strategy="parallel",
#             processing_time_ms={"total_ms": 400},
#             confidence_scores={"overall": globule_data["parsing_confidence"]},
#             interpretations=[],
#             has_nuance=False,
#             semantic_neighbors=[],
#             processing_notes=[],
#             created_at=datetime.fromisoformat(globule_data["created_at"])
#         )
#         await real_storage_manager.store_globule(globule)
    
#     return real_storage_manager


# @pytest.fixture
# async def large_populated_storage(real_storage_manager, large_test_dataset):
#     """Create a storage manager populated with large dataset for performance testing (100k records)."""
#     # Store data in larger batches for better performance with 100k records
#     batch_size = 1000
#     globules_data = large_test_dataset["globules"]
    
#     print(f"Loading {len(globules_data)} records for performance testing...")
    
#     for i in range(0, len(globules_data), batch_size):
#         batch = globules_data[i:i + batch_size]
        
#         for globule_data in batch:
#             globule = ProcessedGlobule(
#                 id=globule_data["id"],
#                 text=globule_data["text"],
#                 embedding=np.array(globule_data["embedding"], dtype=np.float32),
#                 embedding_confidence=globule_data["embedding_confidence"],
#                 parsed_data=globule_data["parsed_data"],
#                 parsing_confidence=globule_data["parsing_confidence"],
#                 file_decision=FileDecision(
#                     semantic_path=Path(globule_data["file_decision"]["semantic_path"]),
#                     filename=globule_data["file_decision"]["filename"],
#                     metadata=globule_data["file_decision"].get("metadata", {}),
#                     confidence=globule_data["file_decision"]["confidence"],
#                     alternative_paths=globule_data["file_decision"].get("alternative_paths", [])
#                 ),
#                 orchestration_strategy="parallel",
#                 processing_time_ms={"total_ms": 400},
#                 confidence_scores={"overall": globule_data["parsing_confidence"]},
#                 interpretations=[],
#                 has_nuance=False,
#                 semantic_neighbors=[],
#                 processing_notes=[],
#                 created_at=datetime.now() - timedelta(days=globule_data["days_ago"])
#             )
#             await real_storage_manager.store_globule(globule)
        
#         # Progress indicator for large dataset
#         if (i + batch_size) % 10000 == 0:
#             print(f"Loaded {min(i + batch_size, len(globules_data))}/{len(globules_data)} records...")
    
#     print(f"Completed loading {len(globules_data)} records for performance testing.")
#     return real_storage_manager


# @pytest.fixture
# def sample_globule():
#     """Create a sample ProcessedGlobule for unit testing."""
#     return ProcessedGlobule(
#         id="test-globule-123",
#         text="This is a test thought for unit testing",
#         embedding=np.random.randn(1024).astype(np.float32),
#         embedding_confidence=0.95,
#         parsed_data={
#             "title": "Test Thought",
#             "category": "testing",
#             "domain": "development",
#             "keywords": ["test", "unit", "development"],
#             "metadata": {"source": "test"}
#         },
#         parsing_confidence=0.90,
#         file_decision=FileDecision(
#             semantic_path=Path("development/testing"),
#             filename="test-thought.md",
#             metadata={"test": True},
#             confidence=0.8,
#             alternative_paths=[]
#         ),
#         orchestration_strategy="parallel",
#         confidence_scores={"embedding": 0.95, "parsing": 0.90},
#         processing_time_ms={"total": 150, "embedding": 80, "parsing": 70},
#         semantic_neighbors=["neighbor-1", "neighbor-2"],
#         processing_notes=["Test note"],
#         created_at=datetime.now(),
#         modified_at=datetime.now()
#     )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with known relationships for testing."""
    # Create a base vector that we can create variations of
    base_vector = np.random.rand(1024).astype(np.float32)
    
    return {
        "similar_1": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),
        "similar_2": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),
        "different_1": np.random.rand(1024).astype(np.float32),
        "different_2": np.random.rand(1024).astype(np.float32),
        "query": base_vector + np.random.normal(0, 0.05, 1024).astype(np.float32)
    }


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test.db"


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=np.random.randn(1024).astype(np.float32))
    provider.embed_batch = AsyncMock(side_effect=lambda texts: [
        np.random.randn(1024).astype(np.float32) for _ in texts
    ])
    provider.get_dimension = MagicMock(return_value=1024)
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_parsing_provider():
    """Create a mock parsing provider."""
    provider = MagicMock()
    provider.parse = AsyncMock(return_value={
        "title": "Test Title",
        "category": "test",
        "domain": "testing",
        "keywords": ["test", "mock"],
        "metadata": {"mock": True}
    })
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real database"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test requiring large dataset"
    )


def check_vec0_available():
    """Check if vec0 extension is available - integration tests REQUIRE this."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Validate that vec0 extension is available for integration tests."""
    # Integration tests MUST have vec0 available - they should fail if not present
    vec0_available = check_vec0_available()
    
    for item in items:
        if "integration" in item.keywords and not vec0_available:
            # Make integration tests fail with clear message rather than skip
            item.add_marker(pytest.mark.xfail(
                reason="Integration tests require vec0 SQLite extension. Install sqlite-vec package.",
                raises=Exception,
                strict=True
            ))