#!/usr/bin/env python3
"""
Test script for Supabase storage setup
Run this to verify your Supabase configuration is working correctly
"""

import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from headspace.services.supabase_storage import SupabaseStorage
    from data_models import Document, Chunk
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_supabase_setup():
    """Test Supabase connection and embedding operations"""
    
    print("=" * 60)
    print("Supabase Setup Test")
    print("=" * 60)
    print()
    
    # Check environment variables
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials!")
        print()
        print("Please set the following environment variables:")
        print("  - SUPABASE_URL")
        print("  - SUPABASE_KEY")
        print()
        print("You can add them to your .env file or export them:")
        print("  export SUPABASE_URL=https://your-project.supabase.co")
        print("  export SUPABASE_KEY=your-anon-key")
        return False
    
    print(f"✅ Found SUPABASE_URL: {supabase_url[:30]}...")
    print(f"✅ Found SUPABASE_KEY: {supabase_key[:20]}...")
    print()
    
    # Initialize storage
    try:
        storage = SupabaseStorage(supabase_url, supabase_key, user_id="test_user")
        print("✅ SupabaseStorage initialized")
    except Exception as e:
        print(f"❌ Failed to initialize SupabaseStorage: {e}")
        return False
    
    # Test connection
    print("\n1. Testing connection...")
    if storage.test_connection():
        print("   ✅ Connection successful!")
    else:
        print("   ❌ Connection failed!")
        return False
    
    # Test document operations
    print("\n2. Testing document operations...")
    test_doc = Document(
        id="test_doc_123",
        title="Test Document",
        content="This is a test document for Supabase setup verification.",
        doc_type="test",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"test": True}
    )
    
    try:
        doc_id = storage.save_document(test_doc)
        print(f"   ✅ Document saved: {doc_id}")
        
        retrieved_doc = storage.get_document(doc_id)
        if retrieved_doc:
            print(f"   ✅ Document retrieved: {retrieved_doc.title}")
        else:
            print("   ❌ Failed to retrieve document")
            return False
    except Exception as e:
        print(f"   ❌ Document operation failed: {e}")
        return False
    
    # Test chunk operations with embedding
    print("\n3. Testing chunk operations with embeddings...")
    
    # Create a test embedding (mock embedding vector)
    test_embedding = np.random.randn(384).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
    
    test_chunk = Chunk(
        id="test_chunk_123",
        document_id=doc_id,
        chunk_index=0,
        content="This is a test chunk with an embedding.",
        chunk_type="test",
        embedding=test_embedding.tolist(),  # Convert to list
        position_3d=[0.0, 0.0, 0.0],
        color="#FF0000",
        metadata={"test": True},
        tags=["test", "setup"],
        tag_confidence={"test": 0.95},
        reasoning="Test chunk for setup verification",
        embedding_model="test-model"
    )
    
    try:
        chunk_id = storage.save_chunk(test_chunk)
        print(f"   ✅ Chunk saved: {chunk_id}")
        print(f"   ✅ Embedding serialized: {len(test_chunk.embedding)} dimensions")
        
        # Retrieve chunks
        chunks = storage.get_chunks_by_document(doc_id)
        if chunks:
            retrieved_chunk = chunks[0]
            print(f"   ✅ Chunk retrieved: {retrieved_chunk.id}")
            
            # Verify embedding
            if retrieved_chunk.embedding:
                print(f"   ✅ Embedding deserialized: {len(retrieved_chunk.embedding)} dimensions")
                
                # Check if embedding values match (within floating point tolerance)
                if len(retrieved_chunk.embedding) == len(test_chunk.embedding):
                    diff = np.abs(np.array(retrieved_chunk.embedding) - np.array(test_chunk.embedding))
                    max_diff = np.max(diff)
                    if max_diff < 1e-6:
                        print(f"   ✅ Embedding values match (max diff: {max_diff:.2e})")
                    else:
                        print(f"   ⚠️  Embedding values differ (max diff: {max_diff:.2e})")
                else:
                    print(f"   ❌ Embedding dimension mismatch!")
                    return False
            else:
                print("   ❌ No embedding found in retrieved chunk")
                return False
        else:
            print("   ❌ Failed to retrieve chunks")
            return False
    except Exception as e:
        print(f"   ❌ Chunk operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup test data
    print("\n4. Cleaning up test data...")
    try:
        storage.delete_document(doc_id)
        print("   ✅ Test data cleaned up")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Supabase is configured correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # Load environment variables from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not required
    
    success = test_supabase_setup()
    sys.exit(0 if success else 1)

