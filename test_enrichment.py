#!/usr/bin/env python3
"""
Quick test script for enrichment system
Tests both small (sync) and large (async) document creation
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_small_document():
    """Test synchronous enrichment for small documents"""
    print("\n" + "="*60)
    print("TEST 1: Small Document (Synchronous Enrichment)")
    print("="*60)
    
    data = {
        "title": "Test Small Document",
        "content": "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.",
        "doc_type": "text"
    }
    
    print(f"📤 Creating document: {data['title']}")
    response = requests.post(f"{BASE_URL}/api/documents", json=data)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    doc_id = result["id"]
    status = result.get("status", "unknown")
    
    print(f"✅ Document created: {doc_id}")
    print(f"📊 Status: {status}")
    
    if status == "enriched":
        print("✅ Small document enriched synchronously (as expected)")
        
        # Verify chunks have embeddings
        doc_response = requests.get(f"{BASE_URL}/api/documents/{doc_id}")
        if doc_response.status_code == 200:
            doc_data = doc_response.json()
            chunks = doc_data.get("chunks", [])
            enriched = sum(1 for c in chunks if c.get("embedding") and len(c.get("embedding", [])) > 0)
            print(f"📊 Chunks with embeddings: {enriched}/{len(chunks)}")
            
            if enriched == len(chunks):
                print("✅ All chunks have embeddings!")
            else:
                print(f"⚠️  Only {enriched}/{len(chunks)} chunks have embeddings")
    else:
        print(f"⚠️  Expected 'enriched' status, got '{status}'")
    
    return doc_id


def test_large_document():
    """Test asynchronous enrichment for large documents"""
    print("\n" + "="*60)
    print("TEST 2: Large Document (Asynchronous Enrichment)")
    print("="*60)
    
    # Create a document with many paragraphs
    paragraphs = [f"This is paragraph number {i+1} with some content." for i in range(15)]
    content = "\n\n".join(paragraphs)
    
    data = {
        "title": "Test Large Document",
        "content": content,
        "doc_type": "text"
    }
    
    print(f"📤 Creating document: {data['title']} ({len(paragraphs)} paragraphs)")
    response = requests.post(f"{BASE_URL}/api/documents", json=data)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    doc_id = result["id"]
    status = result.get("status", "unknown")
    
    print(f"✅ Document created: {doc_id}")
    print(f"📊 Status: {status}")
    
    if status == "processing":
        print("✅ Large document using background enrichment (as expected)")
        
        # Poll status endpoint
        print("\n📊 Polling enrichment status...")
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/api/documents/{doc_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                enriched = status_data["chunks"]["enriched"]
                total = status_data["chunks"]["total"]
                progress = int((enriched / total) * 100) if total > 0 else 0
                
                print(f"  Progress: {enriched}/{total} chunks ({progress}%) - Status: {status_data['status']}")
                
                if status_data["is_enriched"]:
                    print("✅ Enrichment complete!")
                    return doc_id
                
            time.sleep(1)
        
        print(f"⚠️  Enrichment didn't complete within {max_wait} seconds")
    else:
        print(f"⚠️  Expected 'processing' status, got '{status}'")
    
    return doc_id


def test_status_endpoint(doc_id):
    """Test the status endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Status Endpoint")
    print("="*60)
    
    if not doc_id:
        print("⚠️  No document ID provided")
        return
    
    response = requests.get(f"{BASE_URL}/api/documents/{doc_id}/status")
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return
    
    status = response.json()
    print(f"📊 Document ID: {status['document_id']}")
    print(f"📊 Status: {status['status']}")
    print(f"📊 Is Enriched: {status['is_enriched']}")
    print(f"📊 Chunks: {status['chunks']['enriched']}/{status['chunks']['total']} enriched")
    print(f"📊 Pending: {status['chunks']['pending']}")


def main():
    print("\n🧪 Testing Headspace Enrichment System")
    print("="*60)
    print(f"🌐 Base URL: {BASE_URL}")
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=3)
        if health.status_code == 200:
            print("✅ Server is running\n")
        else:
            print(f"⚠️  Health endpoint returned {health.status_code}, but continuing anyway...\n")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running:")
        print("   python headspace/main.py")
        print("\n💡 Starting server in background...")
        print("   (You may need to start it manually in another terminal)")
        sys.exit(1)
    except Exception as e:
        print(f"⚠️  Error checking server: {e}")
        print("   Continuing anyway...\n")
    
    # Run tests
    small_doc_id = test_small_document()
    large_doc_id = test_large_document()
    
    # Test status endpoint
    if large_doc_id:
        test_status_endpoint(large_doc_id)
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)
    print("\n💡 Next steps:")
    print("   1. Check server logs for enrichment progress")
    print("   2. Test WebSocket: ws://localhost:8000/ws/enrichment/{doc_id}")
    print("   3. Check visualization at http://localhost:8000")
    print()


if __name__ == "__main__":
    main()

