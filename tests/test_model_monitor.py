#!/usr/bin/env python3
"""
Test script for the enhanced model monitoring system
"""

import requests
import json
import time

def test_health_endpoints():
    """Test all health check endpoints"""
    base_url = "http://localhost:8000"

    print("\n" + "="*80)
    print("🧪 TESTING MODEL MONITORING SYSTEM")
    print("="*80)

    # Test 1: Basic health check
    print("\n1. Testing basic health check endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['status']}")
            print(f"   Timestamp: {data['timestamp']}")
        else:
            print(f"   ❌ Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to reach health endpoint: {e}")

    # Test 2: Model status endpoint
    print("\n2. Testing model status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model status retrieved successfully")
            print(f"   Overall status: {data['status']}")
            print(f"   Services:")
            for service, available in data['services'].items():
                status_icon = "✓" if available else "✗"
                print(f"      {status_icon} {service}: {'Available' if available else 'Not Available'}")

            if 'models' in data:
                print(f"   Registered Models:")
                for model_id, model_info in data['models'].items():
                    print(f"      📦 {model_id}:")
                    print(f"         Type: {model_info.get('type', 'Unknown')}")
                    print(f"         Provider: {model_info.get('provider', 'Unknown')}")
                    print(f"         Status: {model_info.get('status', 'Unknown')}")
                    metrics = model_info.get('metrics', {})
                    if metrics.get('requests', 0) > 0:
                        success_rate = (metrics.get('successes', 0) / metrics.get('requests', 0)) * 100
                        print(f"         Requests: {metrics.get('requests', 0)} (Success rate: {success_rate:.1f}%)")
                        print(f"         Avg Latency: {metrics.get('average_latency_ms', 0):.2f}ms")
        else:
            print(f"   ❌ Model status failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to reach model status endpoint: {e}")

    # Test 3: Detailed health check
    print("\n3. Testing detailed health check endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health/detailed", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Detailed health check completed")

            if 'services' in data:
                for service, details in data['services'].items():
                    print(f"\n   📋 {service.upper()}:")
                    status = details.get('status', 'Unknown')
                    print(f"      Status: {status}")

                    if 'error' in details:
                        print(f"      Error: {details['error']}")
                    if 'fix' in details:
                        print(f"      Fix: {details['fix']}")
                    if 'available_models' in details:
                        print(f"      Available models: {', '.join(details['available_models'][:3])}...")
                    if 'response_time_ms' in details:
                        print(f"      Response time: {details['response_time_ms']:.2f}ms")
                    if 'embedding_dimension' in details:
                        print(f"      Embedding dimension: {details['embedding_dimension']}")
        else:
            print(f"   ❌ Detailed health check failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to reach detailed health endpoint: {e}")

def test_document_creation_with_monitoring():
    """Test document creation and observe monitoring logs"""
    base_url = "http://localhost:8000"

    print("\n" + "="*80)
    print("4. Testing document creation with monitoring...")
    print("="*80)

    test_document = {
        "title": "Model Monitor Test Document",
        "content": """This is a test document to verify the model monitoring system.

        The monitoring system should track:
        - Chunking operations (LLM or fallback)
        - Embedding generation
        - Tagging operations
        - Latency for each operation
        - Success/failure rates

        Each model operation should be logged with detailed timing information.""",
        "doc_type": "text"
    }

    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/documents",
                                  json=test_document,
                                  timeout=30)

        total_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Document created successfully")
            print(f"   Document ID: {data['id']}")
            print(f"   Total processing time: {total_time:.2f}ms")
            print(f"   Message: {data['message']}")
        else:
            print(f"   ❌ Document creation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed to create document: {e}")

    # Check model status after processing
    print("\n5. Checking model metrics after document processing...")
    try:
        response = requests.get(f"{base_url}/api/health/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                for model_id, model_info in data['models'].items():
                    metrics = model_info.get('metrics', {})
                    if metrics.get('requests', 0) > 0:
                        print(f"   📊 {model_id}:")
                        print(f"      Total requests: {metrics.get('requests', 0)}")
                        print(f"      Successes: {metrics.get('successes', 0)}")
                        print(f"      Failures: {metrics.get('failures', 0)}")
                        print(f"      Average latency: {metrics.get('average_latency_ms', 0):.2f}ms")
    except Exception as e:
        print(f"   ❌ Failed to get updated metrics: {e}")

if __name__ == "__main__":
    print("\nMake sure the headspace system is running on http://localhost:8000")
    print("This test will verify the model monitoring system is working correctly\n")

    test_health_endpoints()
    test_document_creation_with_monitoring()

    print("\n" + "="*80)
    print("✅ MODEL MONITORING TESTS COMPLETE")
    print("Check the server logs for detailed monitoring output")
    print("="*80)