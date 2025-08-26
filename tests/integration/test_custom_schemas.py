"""
Integration tests for enhanced custom schema functionality.

Tests comprehensive valet parsing with mocked Ollama responses for deterministic testing,
plus edge cases and adversarial inputs.
"""

import json
import pytest
import time
from unittest.mock import AsyncMock

from globule.services.parsing.ollama_parser import OllamaParser
from globule.schemas.manager import get_schema_manager


class TestValetSchemaIntegration:
    """Integration tests for the enhanced valet schema system."""

    @pytest.fixture
    async def parser(self):
        """Create parser instance for testing."""
        parser = OllamaParser()
        yield parser
        await parser.close()

    @pytest.fixture
    def schema_manager(self):
        """Get schema manager instance."""
        return get_schema_manager()

    def test_valet_schema_loaded(self, schema_manager):
        """Test that enhanced valet schema is properly loaded and configured."""
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Parking Entry Schema'
        
        # Verify required fields match enhanced specification
        required_fields = valet_schema.get('required', [])
        assert 'valet_name' in required_fields
        assert 'vehicle_make' in required_fields
        assert 'vehicle_model' in required_fields
        assert 'license_plate' in required_fields
        assert 'parking_spot' in required_fields
        assert 'keywords' in required_fields
        assert 'entities' in required_fields
        
        # Verify optional fields with enhanced validation
        properties = valet_schema.get('properties', {})
        assert 'damage_notes' in properties
        assert 'damage_notes' not in required_fields
        assert 'timestamp' in properties
        assert 'timestamp' not in required_fields
        
        # Verify enhanced constraints
        vehicle_make_prop = properties.get('vehicle_make', {})
        assert 'pattern' in vehicle_make_prop
        
        license_plate_prop = properties.get('license_plate', {})
        assert 'pattern' in license_plate_prop
        assert license_plate_prop['pattern'] == "^[A-Z0-9\\-]{6,10}$"

    @pytest.mark.asyncio
    async def test_valet_schema_parsing_success(self, parser, schema_manager, mocker):
        """Test successful valet parsing with mocked Ollama response."""
        # Mock Ollama API response
        mock_response = {
            'title': "Valet Parking Entry",
            'category': "car_arrival",
            'domain': "valet",
            'valet_name': "John D.",
            'vehicle_make': "Toyota",
            'vehicle_model': "Camry",
            'license_plate': "5-SAM-123",
            'parking_spot': "C7",
            'timestamp': None,
            'damage_notes': ["small scratch on rear bumper"],
            'keywords': ["valet", "parked", "Toyota Camry"],
            'entities': ["John D.", "Toyota Camry", "5-SAM-123", "C7"],
            'sentiment': "neutral",
            'content_type': "log_entry",
            'confidence_score': 0.95
        }
        
        # Mock the Ollama API call
        mock_call = AsyncMock(return_value=json.dumps(mock_response))
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        # Test input
        sample_text = ("Valet: Parked a black Toyota Camry, license plate 5-SAM-123, "
                       "in spot C7. Noted a small scratch on the rear bumper. Valet: John D.")
        
        # Parse with valet domain
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify successful parsing
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result['metadata']['parser_version'] == '3.0.0'
        assert result['metadata']['schema_used'] == 'valet'
        
        # Verify extracted values
        assert result.get('valet_name') == 'John D.'
        assert result.get('vehicle_make') == 'Toyota'
        assert result.get('vehicle_model') == 'Camry'
        assert result.get('license_plate') == '5-SAM-123'
        assert result.get('parking_spot') == 'C7'
        assert result.get('timestamp') is None
        
        # Verify damage extraction
        damage_notes = result.get('damage_notes', [])
        assert len(damage_notes) > 0
        assert any('scratch' in note.lower() for note in damage_notes)
        
        # Verify arrays
        assert len(result.get('keywords', [])) > 0
        assert len(result.get('entities', [])) > 0

    @pytest.mark.asyncio
    async def test_valet_parsing_no_damage(self, parser, schema_manager, mocker):
        """Test valet parsing with clean input (no damage)."""
        # Mock clean response
        mock_response = {
            'title': "Clean Parking Entry",
            'category': "car_arrival",
            'domain': "valet",
            'valet_name': "Alice Smith",
            'vehicle_make': "Honda",
            'vehicle_model': "Civic",
            'license_plate': "ABC-123",
            'parking_spot': "A1",
            'timestamp': None,
            'damage_notes': [],
            'keywords': ["parked", "Honda Civic"],
            'entities': ["Alice Smith", "Honda Civic", "ABC-123", "A1"],
            'sentiment': "neutral",
            'content_type': "log_entry",
            'confidence_score': 0.90
        }
        
        mock_call = AsyncMock(return_value=json.dumps(mock_response))
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        # Clean input text
        sample_text = ("Alice Smith parked a Honda Civic with plate ABC-123 in spot A1. "
                       "No issues noted.")
        
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify clean parsing
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result.get('valet_name') == 'Alice Smith'
        assert result.get('vehicle_make') == 'Honda'
        
        # Should have empty damage notes
        damage_notes = result.get('damage_notes', [])
        assert len(damage_notes) == 0

    @pytest.mark.asyncio
    async def test_parsing_retry_logic(self, parser, schema_manager, mocker):
        """Test retry logic with initial failure then success."""
        # Mock first call returns invalid JSON, second succeeds
        valid_response = {
            'title': "Retry Success",
            'category': "car_arrival",
            'domain': "valet",
            'valet_name': "Test User",
            'vehicle_make': "Ford",
            'vehicle_model': "Focus",
            'license_plate': "TEST-123",
            'parking_spot': "B2",
            'keywords': ["test"],
            'entities': ["Test User", "Ford Focus"],
            'sentiment': "neutral",
            'content_type': "log_entry",
            'confidence_score': 0.85
        }
        
        mock_call = AsyncMock(side_effect=[
            "Invalid JSON response",  # First attempt fails
            json.dumps(valid_response)  # Second attempt succeeds
        ])
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        sample_text = "Test User parked Ford Focus, plate TEST-123 in B2."
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Should succeed on retry
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result['metadata']['attempts'] == 2
        assert result.get('valet_name') == 'Test User'

    @pytest.mark.asyncio
    async def test_parsing_failure_fallback(self, parser, schema_manager, mocker):
        """Test fallback behavior when all parsing attempts fail."""
        # Mock all attempts to fail
        mock_call = AsyncMock(side_effect=Exception("API Error"))
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        sample_text = "Some valet text that fails to parse"
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Should fall back gracefully
        assert result['metadata']['parser_type'] == 'fatal_fallback'
        assert 'error' in result
        assert result['title'] == sample_text[:50]

    @pytest.mark.asyncio
    async def test_adversarial_input(self, parser, schema_manager, mocker):
        """Test parsing with adversarial/edge case inputs."""
        # Mock response for adversarial input
        mock_response = {
            'title': "Unclear Entry",
            'category': "note",
            'domain': "valet",
            'valet_name': "Unknown",
            'vehicle_make': "Unknown",
            'vehicle_model': "Unknown",
            'license_plate': "UNK-000",
            'parking_spot': "Unknown",
            'keywords': ["unclear", "incomplete"],
            'entities': [],
            'sentiment': "neutral",
            'content_type': "log_entry",
            'confidence_score': 0.30
        }
        
        mock_call = AsyncMock(return_value=json.dumps(mock_response))
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        # Adversarial input with minimal info
        adversarial_text = "Car? Maybe parked somewhere. Not sure about details."
        result = await parser.parse(adversarial_text, {'domain': 'valet'})
        
        # Should handle gracefully with low confidence
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result.get('confidence_score', 0) < 0.5
        assert 'Unknown' in result.get('valet_name', '')

    @pytest.mark.asyncio
    async def test_pydantic_validation_enforcement(self, parser, schema_manager, mocker):
        """Test that Pydantic validation catches and fixes schema violations."""
        # Mock response with schema violations (extra fields, wrong types)
        invalid_response = {
            'title': "Valid Title",
            'category': "car_arrival",
            'domain': "valet",
            'valet_name': "John Smith",
            'vehicle_make': "Toyota",
            'vehicle_model': "Camry",
            'license_plate': "5-SAM-123",
            'parking_spot': "C7",
            'keywords': ["test"],
            'entities': ["John Smith"],
            'extra_field': "This should be removed",  # Should be filtered out
            'confidence_score': "invalid_type"  # Wrong type, should get default
        }
        
        mock_call = AsyncMock(return_value=json.dumps(invalid_response))
        mocker.patch.object(parser, '_call_ollama_api', mock_call)
        
        sample_text = "John Smith parked Toyota Camry, plate 5-SAM-123 in C7."
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Pydantic should have cleaned up the response
        assert 'extra_field' not in result
        assert isinstance(result.get('confidence_score'), float)  # Should have default
        assert result['metadata']['parser_type'] == 'ollama_llm'

    def test_schema_validation_direct(self, schema_manager):
        """Test direct schema validation with enhanced constraints."""
        # Test valid data
        current_timestamp = int(time.time())
        valid_data = {
            "title": "Valid Entry",
            "category": "car_arrival",
            "domain": "valet",
            "valet_name": "John Smith",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "license_plate": "5-SAM-123",
            "parking_spot": "C7",
            "timestamp": current_timestamp,
            "damage_notes": ["small scratch"],
            "keywords": ["parking", "Toyota"],
            "entities": ["John Smith", "Toyota Camry"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.95
        }
        
        is_valid, errors = schema_manager.validate_data(valid_data, 'valet')
        assert is_valid, f"Valid data failed: {errors}"
        
        # Test invalid license plate pattern
        invalid_data = valid_data.copy()
        invalid_data['license_plate'] = "invalid-plate-format"
        
        is_valid, errors = schema_manager.validate_data(invalid_data, 'valet')
        assert not is_valid
        # Check that validation caught the invalid license plate
        assert any('license_plate' in error.lower() or 'pattern' in error.lower() for error in errors)

    def test_schema_manager_domain_mapping(self, schema_manager):
        """Test that valet domain correctly maps to enhanced valet schema."""
        schema_name = schema_manager.get_schema_for_domain('valet')
        assert schema_name == 'valet'
        
        # Verify the schema exists and loads correctly
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Parking Entry Schema'
        
        # Test alternative domain mapping
        parking_schema = schema_manager.get_schema_for_domain('parking_service')
        assert parking_schema in ['valet', 'default']  # Should map to valet or fallback