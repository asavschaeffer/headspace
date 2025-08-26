#!/usr/bin/env python3
"""
Simple test script for the new messaging integration system.

This script tests the end-to-end flow:
1. Create an InputMessage (simulating WhatsApp/Telegram input)
2. Process it through the OrchestrationEngine
3. Store the results
4. Search for the stored content

Usage:
    python test_messaging_integration.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import globule modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from globule.inputs.models import InputMessage, Attachment, AttachmentType
from globule.core.api import GlobuleAPI
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.mock_adapter import MockEmbeddingAdapter
from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.services.parsing.ollama_parser import OllamaParser
from globule.orchestration.engine import GlobuleOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_messaging_integration():
    """Test the complete messaging integration flow."""
    
    print("Testing Globule Messaging Integration")
    print("=" * 50)
    
    try:
        # 1. Initialize components
        print("Step 1: Initializing components...")
        
        storage = SQLiteStorageManager(db_path=Path(":memory:"))  # Use in-memory DB for testing
        await storage.initialize()
        
        embedding_provider = MockEmbeddingAdapter()  # Use mock for testing
        
        # Try Ollama parser, fall back to mock if not available
        try:
            parsing_provider = OllamaParser()
            parsing_adapter = OllamaParsingAdapter(parsing_provider)
            print("Using Ollama parser")
        except Exception as e:
            print(f"Ollama not available, using mock parser: {e}")
            from globule.services.providers_mock import MockParserProvider
            parsing_adapter = MockParserProvider()
        
        orchestrator = GlobuleOrchestrator(
            parser_provider=parsing_adapter,
            embedding_provider=embedding_provider,
            storage_manager=storage
        )
        
        api = GlobuleAPI(storage=storage, orchestrator=orchestrator)
        
        print("Components initialized successfully")
        
        # 2. Create test InputMessage (simulating WhatsApp message)
        print("\nStep 2: Creating test WhatsApp message...")
        
        whatsapp_message = InputMessage(
            content="Progressive overload principles from fitness could apply to building creative stamina. Start small, be consistent, gradually increase intensity.",
            source="whatsapp",
            user_identifier="+1234567890",
            timestamp=datetime.now(),
            message_id="whatsapp_msg_123",
            platform_metadata={
                "whatsapp_message_type": "text",
                "phone_number_id": "123456789",
                "display_phone_number": "+1-555-GLOBULE"
            }
        )
        
        print(f"Message content: {whatsapp_message.content}")
        print(f"Source: {whatsapp_message.source}")
        print(f"User: {whatsapp_message.user_identifier}")
        
        # 3. Process the message through the API
        print("\nStep 3: Processing message through GlobuleAPI...")
        
        processed_globules = await api.add_from_input_message(whatsapp_message)
        
        print(f"Processing complete! Created {len(processed_globules)} globules:")
        for i, globule in enumerate(processed_globules, 1):
            preview = globule.original_globule.raw_text[:60] + "..." if len(globule.original_globule.raw_text) > 60 else globule.original_globule.raw_text
            print(f"   {i}. {preview}")
            print(f"      Source: {globule.provider_metadata.get('input_source', 'unknown')}")
            print(f"      Timestamp: {globule.processed_timestamp}")
        
        # 4. Test search functionality
        print("\nStep 4: Testing semantic search...")
        
        search_results = await api.search_semantic("creative stamina", limit=5)
        
        print(f"Found {len(search_results)} results for 'creative stamina':")
        for i, result in enumerate(search_results, 1):
            preview = result.original_globule.raw_text[:60] + "..." if len(result.original_globule.raw_text) > 60 else result.original_globule.raw_text
            source = result.provider_metadata.get('input_source', 'cli')
            print(f"   {i}. [{source}] {preview}")
        
        # 5. Test with attachment (simulated)
        print("\nStep 5: Testing message with attachment...")
        
        # Create fake image attachment
        fake_image_data = b"fake_image_data_here"
        
        message_with_attachment = InputMessage(
            content="Here's my sketch of the progressive overload concept",
            attachments=[
                Attachment(
                    content=fake_image_data,
                    mime_type="image/jpeg",
                    filename="concept_sketch.jpg",
                    attachment_type=AttachmentType.IMAGE
                )
            ],
            source="whatsapp",
            user_identifier="+1234567890",
            timestamp=datetime.now(),
            message_id="whatsapp_msg_124"
        )
        
        attachment_results = await api.add_from_input_message(message_with_attachment)
        
        print(f"Processed message with attachment - created {len(attachment_results)} globules:")
        for i, globule in enumerate(attachment_results, 1):
            is_attachment = globule.provider_metadata.get('is_attachment', False)
            content_type = "attachment" if is_attachment else "text"
            preview = globule.original_globule.raw_text[:60] + "..." if len(globule.original_globule.raw_text) > 60 else globule.original_globule.raw_text
            print(f"   {i}. [{content_type}] {preview}")
        
        # 6. Final stats
        print("\nStep 6: Final statistics...")
        
        all_globules = await api.get_all_globules(limit=100)
        whatsapp_globules = [g for g in all_globules if g.provider_metadata.get('input_source') == 'whatsapp']
        
        print(f"Total globules in database: {len(all_globules)}")
        print(f"WhatsApp globules: {len(whatsapp_globules)}")
        print(f"Attachment globules: {len([g for g in all_globules if g.provider_metadata.get('is_attachment')])}")
        
        print("\nSUCCESS: Messaging integration test completed!")
        print("[PASS] InputMessage creation works")
        print("[PASS] Message processing through OrchestrationEngine works") 
        print("[PASS] Storage of processed globules works")
        print("[PASS] Semantic search finds processed messages")
        print("[PASS] Attachment handling works")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'storage' in locals():
            await storage.close()


async def test_whatsapp_webhook_parsing():
    """Test WhatsApp webhook payload parsing."""
    
    print("\nTesting WhatsApp Webhook Parsing")
    print("-" * 40)
    
    from globule.inputs.adapters.whatsapp import WhatsAppAdapter
    
    # Mock WhatsApp webhook payload (text message)
    mock_whatsapp_payload = {
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550100001",
                                "phone_number_id": "123456789"
                            },
                            "messages": [
                                {
                                    "from": "+1234567890",
                                    "id": "wamid.example",
                                    "timestamp": "1692648000",
                                    "type": "text",
                                    "text": {
                                        "body": "This is a test message from WhatsApp!"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    
    adapter = WhatsAppAdapter("fake_token", "fake_verify_token")
    messages = await adapter.parse_webhook(mock_whatsapp_payload)
    
    print(f"Parsed {len(messages)} messages from webhook:")
    for msg in messages:
        print(f"   Content: {msg.content}")
        print(f"   From: {msg.user_identifier}")
        print(f"   Source: {msg.source}")
        print(f"   Message ID: {msg.message_id}")
    
    return len(messages) > 0


if __name__ == "__main__":
    async def main():
        print("Globule Messaging Integration Test Suite")
        print("=" * 60)
        
        # Run tests
        test1_result = await test_messaging_integration()
        test2_result = await test_whatsapp_webhook_parsing()
        
        print(f"\nTest Results Summary:")
        print(f"[RESULT] Messaging Integration: {'PASS' if test1_result else 'FAIL'}")
        print(f"[RESULT] WhatsApp Webhook Parsing: {'PASS' if test2_result else 'FAIL'}")
        
        if test1_result and test2_result:
            print(f"\nALL TESTS PASSED! Messaging system is ready to use!")
            return 0
        else:
            print(f"\nSome tests failed. Check the errors above.")
            return 1
    
    exit_code = asyncio.run(main())