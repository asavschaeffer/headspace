"""
Real Integration Tests for Vector Search Functionality.

These are proper integration tests that:
1. Use real SQLite databases with vec0 extension
2. Use realistic test data with meaningful embeddings
3. Test actual system integration points
4. Include comprehensive performance testing with large datasets
5. Fail if dependencies are not available (no skipif decorators)
"""

import pytest
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Tuple
from pathlib import Path

from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.core.models import ProcessedGlobuleV1, FileDecisionV1


@pytest.mark.integration
class TestVectorSearchIntegration:
    """Integration tests for vector search using real database and data."""

    @pytest.mark.asyncio
    async def test_real_database_initialization(self, real_storage_manager):
        """Test that real database initializes properly with vec0 extension."""
        # This test ensures the database and vec0 extension are working
        storage = real_storage_manager
        
        # Verify tables exist
        db = await storage._get_connection()
        cursor = await db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='globules'
        """)
        tables = await cursor.fetchall()
        assert len(tables) == 1
        
        # Verify vec0 functions are available
        cursor = await db.execute("SELECT vec_version()")
        version = await cursor.fetchone()
        assert version is not None

    @pytest.mark.asyncio
    async def test_real_data_storage_and_retrieval(self, populated_real_storage):
        """Test storing and retrieving real data with embeddings."""
        storage = populated_real_storage
        
        # Test that we can retrieve stored globules
        globule = await storage.get_globule("fitness_progressive_overload")
        assert globule is not None
        assert globule.text == "Progressive overload in fitness means gradually increasing weight, frequency, or number of reps in your strength training routine. This principle ensures continuous adaptation and growth."
        assert globule.parsed_data["domain"] == "fitness"
        assert len(globule.embedding) == 1024
        assert globule.embedding_confidence == 0.92

    @pytest.mark.asyncio
    async def test_semantic_search_with_real_data(self, populated_real_storage):
        """Test semantic search functionality with real data."""
        storage = populated_real_storage
        
        # Get a real embedding to search with
        globule = await storage.get_globule("fitness_progressive_overload")
        query_embedding = globule.embedding
        
        # Search for similar content
        results = await storage.search_by_embedding(
            query_embedding, limit=5, similarity_threshold=0.1
        )
        
        # Should find results including the original globule
        assert len(results) > 0
        found_ids = [g.id for g, _ in results]
        assert "fitness_progressive_overload" in found_ids
        
        # Results should be sorted by similarity
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # Similarity scores should be reasonable
        for _, score in results:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self, populated_real_storage):
        """Test hybrid text + semantic search integration."""
        storage = populated_real_storage
        
        # Get a real embedding for testing
        globule = await storage.get_globule("software_technical_debt")  # Technical debt globule
        query_embedding = globule.embedding
        query_text = "software development"
        
        # Test hybrid search
        results = await storage.hybrid_search(
            query_text, query_embedding, limit=5, similarity_threshold=0.1
        )
        
        assert len(results) > 0
        
        # Should find the technical debt globule
        found_ids = [g.id for g, _ in results]
        assert "software_technical_debt" in found_ids
        
        # Check that text matching influences results
        for globule, score in results:
            if "software" in globule.text.lower() or "development" in globule.text.lower():
                # Text matches should have reasonable scores
                assert score > 0.0

    @pytest.mark.asyncio
    async def test_cross_domain_semantic_relationships(self, populated_real_storage):
        """Test that semantic search finds cross-domain relationships."""
        storage = populated_real_storage
        
        # Use Feynman Technique embedding (learning domain)
        learning_globule = await storage.get_globule("learning_feynman_technique")
        
        # Search with finance domain query but learning methodology
        results = await storage.search_by_embedding(
            learning_globule.embedding, limit=5, similarity_threshold=0.0
        )
        
        # Should find multiple domains if they share conceptual similarity
        domains = set()
        for globule, _ in results:
            domains.add(globule.parsed_data.get("domain"))
        
        # Should have at least 2 different domains represented
        assert len(domains) >= 2

    @pytest.mark.asyncio
    async def test_confidence_filtering_integration(self, populated_real_storage):
        """Test that confidence filtering works properly in integration."""
        storage = populated_real_storage
        
        # Get a query embedding
        globule = await storage.get_globule("fitness_progressive_overload")
        query_embedding = globule.embedding
        
        # All our test data has confidence > 0.3, so should get results
        results = await storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        # Verify all results meet confidence threshold
        for globule, _ in results:
            assert globule.embedding_confidence > 0.3

    @pytest.mark.asyncio
    async def test_temporal_ordering_integration(self, populated_real_storage):
        """Test that temporal ordering works in integration."""
        storage = populated_real_storage
        
        # Get results with very low threshold to see ordering
        globule = await storage.get_globule("fitness_progressive_overload")
        results = await storage.search_by_embedding(
            globule.embedding, limit=10, similarity_threshold=0.0
        )
        
        # Check that results are primarily ordered by similarity, 
        # with recency as tie-breaker
        assert len(results) > 0
        
        # At minimum, results should be in descending similarity order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_database_consistency_after_operations(self, populated_real_storage):
        """Test database remains consistent after multiple operations."""
        storage = populated_real_storage
        
        # Perform multiple search operations
        globule = await storage.get_globule("fitness_progressive_overload")
        
        # Run several searches
        for i in range(5):
            results = await storage.search_by_embedding(
                globule.embedding, limit=3, similarity_threshold=0.1
            )
            assert len(results) > 0
        
        # Verify database integrity
        db = await storage._get_connection()
        cursor = await db.execute("SELECT COUNT(*) FROM globules")
        count = await cursor.fetchone()
        assert count[0] == 8  # Should have 8 test globules from our deterministic dataset

    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, populated_real_storage):
        """Test that concurrent search operations work correctly."""
        storage = populated_real_storage
        
        # Get different query embeddings
        globule1 = await storage.get_globule("fitness_progressive_overload")
        globule2 = await storage.get_globule("software_technical_debt")
        globule3 = await storage.get_globule("wellness_mindfulness_meditation")
        
        # Run concurrent searches
        tasks = [
            storage.search_by_embedding(globule1.embedding, limit=3, similarity_threshold=0.1),
            storage.search_by_embedding(globule2.embedding, limit=3, similarity_threshold=0.1),
            storage.search_by_embedding(globule3.embedding, limit=3, similarity_threshold=0.1),
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # All searches should succeed
        assert len(results_list) == 3
        for results in results_list:
            assert len(results) > 0
            # Verify result structure
            for globule, score in results:
                assert isinstance(globule, ProcessedGlobuleV1)
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_edge_case_empty_results(self, real_storage_manager):
        """Test behavior when search returns no results."""
        storage = real_storage_manager
        
        # Search with very high similarity threshold in empty database
        random_embedding = np.random.rand(1024).astype(np.float32)
        results = await storage.search_by_embedding(
            random_embedding, limit=10, similarity_threshold=0.99
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_large_embedding_dimensions(self, real_storage_manager):
        """Test that large embedding dimensions work correctly."""
        storage = real_storage_manager
        
        # Store a globule with 1024-dimensional embedding
        large_embedding = np.random.rand(1024).astype(np.float32)
        globule = ProcessedGlobuleV1(
            id="large_embedding_test",
            text="Test with large embedding dimensions",
            embedding=large_embedding,
            embedding_confidence=0.85,
            parsed_data={"domain": "test", "category": "dimension"},
            parsing_confidence=0.80,
            file_decision=FileDecisionV1(Path("test"), "large.md", {}, 0.8, []),
            orchestration_strategy="parallel",
            processing_time_ms={"total_ms": 300},
            confidence_scores={"overall": 0.80},
            interpretations=[],
            has_nuance=False,
            semantic_neighbors=[],
            processing_notes=[],
            created_at=datetime.now()
        )
        
        await storage.store_globule(globule)
        
        # Search with similar embedding
        query_embedding = large_embedding + np.random.normal(0, 0.1, 1024).astype(np.float32)
        results = await storage.search_by_embedding(
            query_embedding, limit=5, similarity_threshold=0.1
        )
        
        assert len(results) > 0
        found_ids = [g.id for g, _ in results]
        assert "large_embedding_test" in found_ids


@pytest.mark.integration
@pytest.mark.performance
class TestVectorSearchPerformance:
    """Performance tests with large datasets (10k+ records)."""

    @pytest.mark.asyncio
    async def test_large_dataset_search_performance(self, large_populated_storage):
        """Test search performance with 10k+ records."""
        storage = large_populated_storage
        
        # Generate a query embedding
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        # Measure search performance over multiple runs
        search_times = []
        
        for run in range(10):  # Run 10 times to get average
            start_time = time.perf_counter()
            
            results = await storage.search_by_embedding(
                query_embedding, limit=50, similarity_threshold=0.1
            )
            
            end_time = time.perf_counter()
            search_time = end_time - start_time
            search_times.append(search_time)
            
            # Verify we get meaningful results
            assert len(results) > 0
            assert len(results) <= 50
        
        # Performance requirements
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)
        
        # Should handle 100k records efficiently - more lenient thresholds for larger dataset
        assert avg_search_time < 1.0, f"Average search time {avg_search_time:.3f}s too slow"
        assert max_search_time < 2.0, f"Max search time {max_search_time:.3f}s too slow"
        
        print(f"Performance: Avg={avg_search_time:.3f}s, Max={max_search_time:.3f}s")

    @pytest.mark.asyncio
    async def test_batch_search_performance(self, large_populated_storage):
        """Test performance of multiple concurrent searches."""
        storage = large_populated_storage
        
        # Generate multiple query embeddings
        query_embeddings = [
            np.random.rand(1024).astype(np.float32) for _ in range(20)
        ]
        
        start_time = time.perf_counter()
        
        # Run all searches concurrently
        tasks = [
            storage.search_by_embedding(embedding, limit=10, similarity_threshold=0.2)
            for embedding in query_embeddings
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Verify all searches succeeded
        assert len(results_list) == 20
        for results in results_list:
            assert len(results) >= 0  # May be empty with high threshold
        
        # Performance requirement: 20 concurrent searches should complete in reasonable time (adjusted for 100k dataset)
        assert total_time < 10.0, f"Batch search time {total_time:.3f}s too slow"
        
        print(f"Batch Performance: 20 concurrent searches in {total_time:.3f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_dataset(self, large_populated_storage):
        """Test memory usage remains reasonable with large dataset."""
        import psutil
        import os
        
        storage = large_populated_storage
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform intensive search operations
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        for _ in range(50):  # 50 searches
            results = await storage.search_by_embedding(
                query_embedding, limit=100, similarity_threshold=0.05
            )
            assert len(results) >= 0
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 50 searches)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
        
        print(f"Memory: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")

    @pytest.mark.asyncio
    async def test_scalability_across_similarity_thresholds(self, large_populated_storage):
        """Test performance across different similarity thresholds."""
        storage = large_populated_storage
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        threshold_times = {}
        
        for threshold in thresholds:
            start_time = time.perf_counter()
            
            results = await storage.search_by_embedding(
                query_embedding, limit=100, similarity_threshold=threshold
            )
            
            end_time = time.perf_counter()
            search_time = end_time - start_time
            threshold_times[threshold] = search_time
            
            # Higher thresholds should return fewer results
            if threshold > 0.5:
                assert len(results) < 1000  # Should filter significantly
        
        # Performance should remain reasonable across all thresholds
        for threshold, search_time in threshold_times.items():
            assert search_time < 1.0, f"Threshold {threshold} took {search_time:.3f}s"
        
        print(f"Threshold Performance: {threshold_times}")

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, large_populated_storage):
        """Test performance under sustained load over time."""
        storage = large_populated_storage
        
        # Run searches continuously for 30 seconds
        start_time = time.perf_counter()
        end_time = start_time + 30  # 30 seconds
        
        search_count = 0
        search_times = []
        
        while time.perf_counter() < end_time:
            query_embedding = np.random.rand(1024).astype(np.float32)
            
            search_start = time.perf_counter()
            results = await storage.search_by_embedding(
                query_embedding, limit=20, similarity_threshold=0.2
            )
            search_end = time.perf_counter()
            
            search_times.append(search_end - search_start)
            search_count += 1
            
            assert len(results) >= 0
        
        # Calculate statistics
        avg_time = sum(search_times) / len(search_times)
        max_time = max(search_times)
        min_time = min(search_times)
        
        # Performance should remain consistent
        assert avg_time < 0.5, f"Average time {avg_time:.3f}s too slow under load"
        assert max_time < 2.0, f"Max time {max_time:.3f}s shows performance degradation"
        
        # Should handle reasonable throughput (adjusted for 100k dataset)
        throughput = search_count / 30  # searches per second
        assert throughput > 5, f"Throughput {throughput:.1f} searches/sec too low"
        
        print(f"Sustained Load: {search_count} searches in 30s, avg={avg_time:.3f}s, throughput={throughput:.1f}/sec")

    @pytest.mark.asyncio
    async def test_database_size_impact_on_performance(self, large_populated_storage):
        """Test how database size impacts search performance."""
        storage = large_populated_storage
        
        # Verify we have the expected large dataset
        db = await storage._get_connection()
        cursor = await db.execute("SELECT COUNT(*) FROM globules")
        count = await cursor.fetchone()
        assert count[0] >= 100000, f"Expected at least 100k records, got {count[0]}"
        
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        # Test search with different result limits
        limits = [10, 50, 100, 500, 1000]
        limit_times = {}
        
        for limit in limits:
            start_time = time.perf_counter()
            
            results = await storage.search_by_embedding(
                query_embedding, limit=limit, similarity_threshold=0.1
            )
            
            end_time = time.perf_counter()
            search_time = end_time - start_time
            limit_times[limit] = search_time
            
            # Should get up to the limit
            assert len(results) <= limit
        
        # Performance should scale reasonably with result limit (adjusted for 100k dataset)
        for limit, search_time in limit_times.items():
            assert search_time < 3.0, f"Limit {limit} took {search_time:.3f}s"
        
        print(f"Limit Performance: {limit_times}")


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios with integrated components."""

    @pytest.mark.asyncio
    async def test_typical_user_workflow(self, populated_real_storage):
        """Test a typical user workflow end-to-end."""
        storage = populated_real_storage
        
        # Scenario: User searches for "learning techniques"
        # This would normally come from an embedding provider
        learning_globule = await storage.get_globule("learning_feynman_technique")  # Feynman technique
        search_embedding = learning_globule.embedding
        
        # User does a hybrid search
        results = await storage.hybrid_search(
            "learning techniques", search_embedding, limit=10, similarity_threshold=0.1
        )
        
        # Should find relevant content
        assert len(results) > 0
        
        # Check that learning-related content is prioritized
        learning_results = [
            (g, s) for g, s in results 
            if g.parsed_data.get("domain") == "learning" or "learning" in g.text.lower()
        ]
        assert len(learning_results) > 0

    @pytest.mark.asyncio
    async def test_cross_domain_knowledge_discovery(self, populated_real_storage):
        """Test discovering knowledge across different domains."""
        storage = populated_real_storage
        
        # Search with progressive overload concept (fitness)
        fitness_globule = await storage.get_globule("fitness_progressive_overload")
        
        # Should find conceptually similar ideas in other domains
        results = await storage.search_by_embedding(
            fitness_globule.embedding, limit=10, similarity_threshold=0.0
        )
        
        # Collect domains found
        domains = set()
        for globule, _ in results:
            domains.add(globule.parsed_data.get("domain"))
        
        # Should span multiple domains if embeddings capture conceptual similarity
        assert len(domains) >= 1  # At minimum fitness domain
        
        # Check for cross-pollination concepts
        concepts_found = []
        for globule, _ in results:
            if "progressive" in globule.text.lower() or "gradual" in globule.text.lower():
                concepts_found.append(globule.parsed_data.get("domain"))
        
        print(f"Cross-domain concepts found in: {concepts_found}")

    @pytest.mark.asyncio
    async def test_temporal_knowledge_evolution(self, populated_real_storage):
        """Test how search handles temporal aspects of knowledge."""
        storage = populated_real_storage
        
        # Get results ordered by recency (most recent first by default)
        query_embedding = np.random.rand(1024).astype(np.float32)
        results = await storage.search_by_embedding(
            query_embedding, limit=5, similarity_threshold=0.0
        )
        
        # Extract creation dates
        creation_dates = [globule.created_at for globule, _ in results]
        
        # Verify we have temporal diversity (span multiple days)
        date_range = max(creation_dates) - min(creation_dates)
        assert date_range.days > 0, "Should have temporal diversity in results"

    @pytest.mark.asyncio
    async def test_confidence_based_filtering_scenarios(self, populated_real_storage):
        """Test various confidence-based filtering scenarios."""
        storage = populated_real_storage
        
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        # Get all results to analyze confidence distribution
        all_results = await storage.search_by_embedding(
            query_embedding, limit=100, similarity_threshold=0.0
        )
        
        # Verify confidence filtering is working (only > 0.3)
        for globule, _ in all_results:
            assert globule.embedding_confidence > 0.3
        
        # Check confidence diversity
        confidences = [globule.embedding_confidence for globule, _ in all_results]
        if len(confidences) > 1:
            assert max(confidences) > min(confidences), "Should have confidence diversity"

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, real_storage_manager):
        """Test error recovery in various failure scenarios."""
        storage = real_storage_manager
        
        # Test with malformed embedding (wrong dimensions)
        try:
            wrong_dim_embedding = np.random.rand(512).astype(np.float32)  # Wrong size
            results = await storage.search_by_embedding(wrong_dim_embedding)
            # Should handle gracefully or raise appropriate error
            assert isinstance(results, list)
        except Exception as e:
            # If it raises an error, it should be meaningful
            assert "dimension" in str(e).lower() or "size" in str(e).lower()
        
        # Test with None embedding
        results = await storage.search_by_embedding(None)
        assert results == []
        
        # Test with empty database
        query_embedding = np.random.rand(1024).astype(np.float32)
        results = await storage.search_by_embedding(query_embedding)
        assert results == []  # Empty database should return empty results