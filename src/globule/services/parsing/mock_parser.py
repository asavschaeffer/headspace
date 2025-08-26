"""
Mock parser for Globule walking skeleton.

Returns hard-coded, empty results for Phase 1 testing.
Will be replaced by real OllamaParser in Phase 2.
"""

from typing import Dict, Any, Optional
import asyncio

from globule.core.interfaces import ParsingProvider


class MockOllamaParser(ParsingProvider):
    """Mock implementation of ParsingProvider for testing"""
    
    async def parse(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return hard-coded empty parsing results"""
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Return minimal parsed data structure
        return {
            "title": text[:50] + "..." if len(text) > 50 else text,
            "category": "note",
            "domain": "general", 
            "keywords": [],
            "entities": [],
            "metadata": {
                "mock": True,
                "parser_version": "mock-0.1.0"
            }
        }