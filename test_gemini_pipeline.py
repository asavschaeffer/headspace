#!/usr/bin/env python3
"""
Comprehensive test script for Headspace Gemini embeddings pipeline.

Tests:
1. Configuration loading
2. Gemini API connectivity and key validation
3. Document creation with simple paragraphs
4. Embedding generation via Gemini
5. 3D position calculation
6. Supabase storage (if configured)
7. Data integrity verification

Run: python test_gemini_pipeline.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_gemini_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test 1: Load and verify configuration"""
    logger.info("=" * 70)
    logger.info("TEST 1: Configuration Loading")
    logger.info("=" * 70)

    try:
        from config_manager import ConfigManager
        config_mgr = ConfigManager()

        logger.info("✅ ConfigManager loaded successfully")

        # Check embedding config
        emb_config = config_mgr.get_embedding_config()
        logger.info(f"   Provider: {emb_config.provider.value}")
        logger.info(f"   Model: {emb_config.name}")
        logger.info(f"   URL: {emb_config.url}")
        logger.info(f"   API Key present: {bool(emb_config.api_key)}")

        # Print config status
        config_mgr.print_status()

        return True
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}", exc_info=True)
        return False


def test_gemini_api_connectivity():
    """Test 2: Test Gemini API connectivity and key validation"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Gemini API Connectivity")
    logger.info("=" * 70)

    try:
        import requests
        from config_manager import ConfigManager

        config_mgr = ConfigManager()
        emb_config = config_mgr.get_embedding_config("gemini")

        if not emb_config.api_key:
            logger.error("❌ Gemini API key not set")
            return False

        logger.info(f"API Key length: {len(emb_config.api_key)} chars")

        # Test with minimal request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{emb_config.name}:batchEmbedContents?key={emb_config.api_key}"

        payload = {
            "requests": [
                {
                    "model": f"models/{emb_config.name}",
                    "content": {"parts": [{"text": "test"}]}
                }
            ]
        }

        logger.info(f"Sending test request to Gemini API...")
        logger.debug(f"URL: {url.replace(emb_config.api_key, '***MASKED***')}")

        response = requests.post(url, json=payload, timeout=30)

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'embeddings' in result and len(result['embeddings']) > 0:
                embedding = result['embeddings'][0].get('values', [])
                logger.info(f"✅ Gemini API working! Got embedding with dimension: {len(embedding)}")
                return True
            else:
                logger.error(f"❌ Unexpected response format: {json.dumps(result)[:200]}")
                return False
        else:
            logger.error(f"❌ API Error {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        logger.error(f"❌ Gemini API test failed: {e}", exc_info=True)
        return False


def test_embedding_engine():
    """Test 3: Test EmbeddingEngine with Gemini"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Embedding Engine")
    logger.info("=" * 70)

    try:
        from config_manager import ConfigManager
        from embeddings_engine import EmbeddingEngine

        config_mgr = ConfigManager()
        engine = EmbeddingEngine(config_mgr)

        test_texts = [
            "This is the first paragraph about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]

        logger.info(f"Testing embedding generation for {len(test_texts)} texts...")

        embeddings = engine.generate_embeddings(test_texts)

        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"   Dimension: {embeddings[0].shape}")
        logger.info(f"   First embedding (first 5 dims): {embeddings[0][:5]}")

        return True

    except Exception as e:
        logger.error(f"❌ Embedding engine test failed: {e}", exc_info=True)
        return False


def test_simple_document_processing():
    """Test 4: Process a simple document with embeddings"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Simple Document Processing")
    logger.info("=" * 70)

    try:
        from config_manager import ConfigManager
        from embeddings_engine import EmbeddingEngine
        from data_models import Document, Chunk
        import numpy as np

        config_mgr = ConfigManager()
        embedder = EmbeddingEngine(config_mgr)

        # Create simple test document
        title = "Quantum Computing Basics"
        content = """Quantum computers use quantum bits or qubits. Unlike classical bits, qubits can exist in a superposition of states. Quantum entanglement allows qubits to be correlated. Quantum gates perform operations on qubits. The measurement of a quantum system collapses its state."""

        # Split into paragraphs (chunks)
        paragraphs = [p.strip() for p in content.split('. ') if p.strip()]

        logger.info(f"Document: '{title}'")
        logger.info(f"Paragraphs: {len(paragraphs)}")

        for i, para in enumerate(paragraphs):
            logger.info(f"  {i+1}. {para[:60]}...")

        # Generate embeddings
        logger.info(f"\nGenerating embeddings for {len(paragraphs)} paragraphs...")
        embeddings = embedder.generate_embeddings(paragraphs)

        logger.info(f"✅ Generated embeddings: {embeddings.shape}")

        # Calculate 3D positions using PCA
        logger.info(f"Calculating 3D positions...")
        from sklearn.decomposition import PCA

        if len(embeddings) > 1:
            pca = PCA(n_components=3)
            positions_3d = pca.fit_transform(embeddings)

            # Normalize to viewable range
            min_pos = positions_3d.min()
            max_pos = positions_3d.max()
            if max_pos > min_pos:
                positions_3d = (positions_3d - min_pos) / (max_pos - min_pos) * 100 - 50
        else:
            positions_3d = np.array([[0, 0, 0]])

        logger.info(f"✅ 3D Positions calculated")
        for i, pos in enumerate(positions_3d):
            logger.info(f"  Chunk {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        return True

    except Exception as e:
        logger.error(f"❌ Document processing test failed: {e}", exc_info=True)
        return False


def test_supabase_storage():
    """Test 5: Test Supabase storage (if configured)"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Supabase Storage")
    logger.info("=" * 70)

    try:
        from config_manager import ConfigManager
        from supabase import create_client

        config_mgr = ConfigManager()

        # Check if Supabase is configured
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not supabase_url or not supabase_key:
            logger.warning("⚠️  Supabase not configured (SUPABASE_URL or SUPABASE_KEY not set)")
            logger.info("   Skipping Supabase tests")
            return True  # Not an error, just skipped

        logger.info(f"Supabase URL: {supabase_url}")

        # Create client
        client = create_client(supabase_url, supabase_key)

        # Test connection
        logger.info("Testing Supabase connection...")
        result = client.table("documents").select("id").limit(1).execute()

        logger.info(f"✅ Supabase connection working!")
        logger.info(f"   Response: {len(result.data)} records")

        return True

    except Exception as e:
        logger.error(f"❌ Supabase test failed: {e}", exc_info=True)
        return False


def test_data_integrity():
    """Test 6: Verify data integrity (no double JSON encoding)"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Data Integrity Check")
    logger.info("=" * 70)

    try:
        from services.supabase_storage import SupabaseStorage
        from data_models import Chunk
        import json

        # Create a test chunk
        chunk = Chunk(
            id="test_chunk_001",
            document_id="test_doc_001",
            chunk_index=0,
            content="This is test content",
            chunk_type="paragraph",
            embedding=[0.1, 0.2, 0.3],
            position_3d=[10.0, 20.0, 30.0],
            color="#ff0000",
            metadata={"source": "test", "status": "complete"},
            tags=["test", "verification"],
            tag_confidence={"test": 0.95},
            reasoning="Testing data integrity"
        )

        logger.info("Testing serialization...")

        # Test serialization
        storage = SupabaseStorage("dummy_url", "dummy_key")  # Won't actually connect
        serialized = storage._serialize_embedding(chunk.embedding)

        logger.info(f"Serialized embedding: {serialized}")

        # Check that it's not double-encoded
        if serialized.startswith('"') or serialized.startswith('['):
            # Try to parse it
            try:
                parsed = json.loads(serialized)
                if isinstance(parsed, str):
                    logger.error("❌ Double JSON encoding detected!")
                    logger.info(f"   First parse: {type(parsed)}")
                    logger.info(f"   Value: {parsed[:100]}")
                    return False
                else:
                    logger.info(f"✅ Correct serialization: {type(parsed)}")
                    return True
            except:
                logger.error("❌ Failed to parse serialized embedding")
                return False
        else:
            logger.error("❌ Unexpected serialization format")
            return False

    except Exception as e:
        logger.error(f"❌ Data integrity test failed: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all tests and report results"""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 10 + "HEADSPACE GEMINI EMBEDDING PIPELINE TEST SUITE" + " " * 12 + "║")
    logger.info("╚" + "=" * 68 + "╝")

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Gemini API Connectivity", test_gemini_api_connectivity),
        ("Embedding Engine", test_embedding_engine),
        ("Document Processing", test_simple_document_processing),
        ("Supabase Storage", test_supabase_storage),
        ("Data Integrity", test_data_integrity),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}", exc_info=True)
            results[name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {name}")

    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready for production use.")
        return 0
    else:
        logger.info("⚠️  Some tests failed. Review logs above for details.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
