"""
Unit tests for the Pydantic data contracts in core.models.
"""
import pytest
import json
from uuid import UUID
from pydantic import ValidationError

from globule.core.models import GlobuleV1, ProcessedGlobuleV1, FileDecisionV1

def test_globule_v1_creation():
    """Tests successful creation of a GlobuleV1 model with default values."""
    globule = GlobuleV1(raw_text="Test input", source="test")
    assert globule.contract_version == '1.0'
    assert isinstance(globule.globule_id, UUID)
    assert globule.raw_text == "Test input"
    assert globule.source == "test"
    assert globule.initial_context == {}

def test_globule_v1_serialization():
    """Tests that GlobuleV1 can be serialized to a dictionary and JSON."""
    globule = GlobuleV1(raw_text="Test input", source="test")
    globule_dict = globule.dict()
    assert globule_dict['raw_text'] == "Test input"
    assert globule_dict['contract_version'] == '1.0'
    
    # Test JSON serialization
    try:
        json.dumps(globule.json())
    except Exception as e:
        pytest.fail(f"JSON serialization failed: {e}")

def test_processed_globule_v1_creation():
    """Tests successful creation of a ProcessedGlobuleV1 model."""
    raw_globule = GlobuleV1(raw_text="Test input", source="test")
    processed_globule = ProcessedGlobuleV1(
        globule_id=raw_globule.globule_id,
        original_globule=raw_globule,
        embedding=[0.1, 0.2, 0.3],
        parsed_data={"key": "value"},
        file_decision=FileDecisionV1(semantic_path="/test", filename="test.md", confidence=0.9),
        processing_time_ms=123.45
    )
    assert processed_globule.contract_version == '1.0'
    assert processed_globule.globule_id == raw_globule.globule_id
    assert processed_globule.original_globule.raw_text == "Test input"
    assert processed_globule.embedding == [0.1, 0.2, 0.3]
    assert processed_globule.file_decision.confidence == 0.9

def test_validation_error():
    """Tests that Pydantic raises a ValidationError for invalid data."""
    with pytest.raises(ValidationError):
        # raw_text is a required field
        GlobuleV1(source="test")
        
    with pytest.raises(ValidationError):
        # Confidence must be between 0.0 and 1.0
        FileDecisionV1(semantic_path="/test", filename="test.md", confidence=1.1)
