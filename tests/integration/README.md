# Integration Tests

## Overview

This directory contains **real integration tests** that properly validate the Globule vector search functionality using actual databases and realistic data.

## Key Improvements

### What Was Wrong Before
- Tests used in-memory databases and mock data
- Tests were skipped if vec0 extension wasn't available
- Performance tests used tiny datasets with arbitrary time limits
- Tests provided false sense of security

### What's Fixed Now
1. **Real Database Usage**: Tests use persistent SQLite databases with actual vec0 extension
2. **Real Data**: Tests use realistic data with meaningful embeddings and content
3. **No Skip Decorators**: Tests fail if dependencies aren't available (as they should)
4. **Proper Performance Testing**: Tests use 10k+ record datasets with meaningful metrics
5. **Integration Testing**: Tests validate actual system integration points

## Test Structure

### TestVectorSearchIntegration
Basic integration tests that validate:
- Real database initialization with vec0 extension
- Storage and retrieval of real data
- Semantic search with meaningful embeddings
- Hybrid text + semantic search
- Cross-domain semantic relationships
- Confidence filtering
- Temporal ordering
- Database consistency
- Concurrent operations
- Edge cases and error handling

### TestVectorSearchPerformance
Performance tests with large datasets (10k+ records):
- Search performance with large datasets
- Batch search performance (concurrent operations)
- Memory usage monitoring
- Scalability across similarity thresholds
- Sustained load testing (30-second runs)
- Database size impact analysis

### TestRealWorldScenarios
Real-world usage scenarios:
- Typical user workflows
- Cross-domain knowledge discovery
- Temporal knowledge evolution
- Confidence-based filtering scenarios
- Error recovery testing

## Fixtures

### Real Database Fixtures
- `real_storage_manager`: Creates temporary real database with vec0
- `populated_real_storage`: Database with realistic test data (5 globules)
- `large_populated_storage`: Database with performance dataset (10k+ globules)

### Test Data
- `real_test_data`: Realistic globules covering fitness, software, wellness, learning, finance
- `large_test_dataset`: 10k+ globules across 8 domains for performance testing

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -m integration

# Run performance tests (these take longer)
pytest tests/integration/ -m performance

# Run specific test class
pytest tests/integration/test_vector_search.py::TestVectorSearchIntegration

# Run with verbose output
pytest tests/integration/ -v -s
```

## Requirements

These tests **require**:
- `sqlite-vec` package installed
- vec0 SQLite extension available
- At least 2GB RAM for large dataset tests
- Tests will **fail** (not skip) if dependencies are missing

## Performance Expectations

Performance tests enforce these requirements:
- Average search time < 0.5s for 10k records
- Max search time < 1.0s for single searches
- Batch searches (20 concurrent) < 5.0s total
- Memory usage increase < 100MB for 50 searches
- Sustained load: > 10 searches/second for 30 seconds

## Data Quality

Test data includes:
- Real, meaningful text content
- Pre-computed embeddings (not random)
- Diverse domains (fitness, software, wellness, learning, finance)
- Realistic confidence scores
- Temporal diversity (different creation dates)

This ensures tests validate actual system behavior rather than artificial scenarios.