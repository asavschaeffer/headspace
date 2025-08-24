"""
Core Pydantic models for Globule, serving as versioned data contracts.
"""
from pydantic import BaseModel, Field, constr, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from datetime import datetime

class BaseGlobuleModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Version 1 Contracts

class NuanceMetaDataV1(BaseGlobuleModel):
    """Metadata capturing nuances in text."""
    has_sarcasm: bool = False
    has_metaphor: bool = False
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0)

class FileDecisionV1(BaseGlobuleModel):
    """Represents a file organization recommendation."""
    semantic_path: str
    filename: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    alternative_paths: List[Any] = Field(default_factory=list)

class GlobuleV1(BaseGlobuleModel):
    """
    Represents a single, unprocessed unit of information captured by the user.
    This is the raw input to the orchestration engine.
    """
    contract_version: Literal['1.0'] = '1.0'
    globule_id: UUID = Field(default_factory=uuid4)
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    raw_text: str
    source: str  # e.g., 'tui', 'cli', 'api'
    
    # Optional context provided at capture time
    initial_context: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingResult(BaseGlobuleModel):
    """
    Contract for embedding service results.
    Ensures consistency and type safety for all embedding operations.
    """
    embedding: List[float] = Field(..., description="The vector embedding")
    dimensions: int = Field(..., gt=0, description="Number of dimensions in the embedding")
    model_name: str = Field(..., description="Name of the model used for embedding")
    processing_time_ms: float = Field(..., ge=0.0, description="Time taken to generate embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific metadata")

class ProcessedContent(BaseGlobuleModel):
    """
    Contract for processor outputs - structured data from content analysis.
    
    This model ensures consistency across different processor types (text, image, audio, etc.)
    while allowing flexible structured data extraction.
    """
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="Extracted structured information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content-specific metadata (e.g., EXIF for images)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Processing confidence score")
    processor_type: str = Field(..., description="Type of processor that generated this content")
    processing_time_ms: float = Field(..., ge=0.0, description="Time taken to process content")

class StructuredQuery(BaseGlobuleModel):
    """
    Contract for structured queries against storage manager.
    
    Enables high-performance queries for specific domains (e.g., valet workflow)
    by querying indexed fields directly rather than full-text or vector search.
    """
    domain: str = Field(..., description="Query domain (e.g., 'valet', 'image', 'task')")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Field-specific filters")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    sort_by: Optional[str] = Field(None, description="Field to sort results by")
    sort_desc: bool = Field(True, description="Sort in descending order")

class EnrichedInput(BaseGlobuleModel):
    """
    Represents enriched input to the orchestration engine.
    Contains the original text along with additional context and metadata.
    """
    original_text: str = Field(..., description="The original raw text")
    enriched_text: str = Field(..., description="The enriched/processed text")
    detected_schema_id: Optional[str] = Field(None, description="Detected schema ID if any")
    schema_config: Optional[Dict[str, Any]] = Field(None, description="Schema configuration")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    source: str = Field(..., description="Source of the input (e.g., 'cli', 'tui', 'api')")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of input")
    verbosity: str = Field("concise", description="Verbosity level for processing")

class ProcessedGlobuleV1(BaseGlobuleModel):
    """
    Represents a Globule after it has been processed by the orchestration engine.
    This is the core data structure used throughout the system.
    """
    contract_version: Literal['1.0'] = '1.0'
    globule_id: UUID
    processed_timestamp: datetime = Field(default_factory=datetime.utcnow)

    original_globule: GlobuleV1
    
    # Enriched data
    embedding: List[float]
    parsed_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis and metadata
    nuances: NuanceMetaDataV1 = Field(default_factory=NuanceMetaDataV1)
    file_decision: Optional[FileDecisionV1] = None
    
    # System metadata
    processing_time_ms: float
    provider_metadata: Dict[str, Any] = Field(default_factory=dict) # Info from parser, embedder, etc.
    
    @property
    def id(self) -> str:
        """Compatibility property for storage manager that expects 'id' field."""
        return str(self.globule_id)
    
    @id.setter
    def id(self, value: str) -> None:
        """Compatibility setter for storage manager that expects 'id' field."""
        self.globule_id = UUID(value)
    
    @property
    def created_at(self) -> datetime:
        """Compatibility property for storage manager that expects 'created_at' field."""
        return self.original_globule.creation_timestamp
    
    @property
    def modified_at(self) -> datetime:
        """Compatibility property for storage manager that expects 'modified_at' field."""
        return self.processed_timestamp
    
    @property
    def text(self) -> str:
        """Compatibility property for storage manager that expects 'text' field."""
        return self.original_globule.raw_text
    
    @property
    def embedding_confidence(self) -> float:
        """Compatibility property for storage manager that expects 'embedding_confidence' field."""
        return self.provider_metadata.get('embedding_confidence', 0.8)
    
    @property
    def parsing_confidence(self) -> float:
        """Compatibility property for storage manager that expects 'parsing_confidence' field."""
        return self.provider_metadata.get('parsing_confidence', 0.8)
    
    @property
    def orchestration_strategy(self) -> str:
        """Compatibility property for storage manager that expects 'orchestration_strategy' field."""
        return self.provider_metadata.get('orchestration_strategy', 'default')
    
    @property
    def confidence_scores(self) -> Dict[str, Any]:
        """Compatibility property for storage manager that expects 'confidence_scores' field."""
        return self.provider_metadata.get('confidence_scores', {})
    
    @property
    def semantic_neighbors(self) -> List[str]:
        """Compatibility property for storage manager that expects 'semantic_neighbors' field."""
        return self.provider_metadata.get('semantic_neighbors', [])
    
    @property
    def processing_notes(self) -> List[str]:
        """Compatibility property for storage manager that expects 'processing_notes' field."""
        return self.provider_metadata.get('processing_notes', [])
    
    @property
    def is_managed(self) -> bool:
        """Returns True if this globule is managed by Globule (has file_decision)."""
        return self.file_decision is not None
    
    @property
    def is_indexed_only(self) -> bool:
        """Returns True if this globule is indexed but not managed (no file_decision)."""
        return self.file_decision is None
    
    @property
    def original_file_path(self) -> Optional[str]:
        """Returns the original file path if this globule was indexed from a file."""
        return self.provider_metadata.get('original_file_path')
    
    @property
    def management_status(self) -> str:
        """Returns a string describing the management status of this globule."""
        if self.is_managed:
            return "managed"
        elif self.original_file_path:
            return "indexed"
        else:
            return "captured"
