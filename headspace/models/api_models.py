"""
API Models for Headspace System
Pydantic models for request/response validation
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator


class ChunkResponse(BaseModel):
    """Response model for chunk data"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    chunk_type: str
    position_3d: List[float] = Field(default_factory=list)
    color: str
    tags: List[str] = Field(default_factory=list)
    reasoning: str
    shape_3d: str
    embedding: List[float] = Field(default_factory=list)  # Embedding vector for procedural geometry
    metadata: Dict = Field(default_factory=dict)
    cluster_id: Optional[int] = None
    cluster_confidence: Optional[float] = None
    cluster_label: Optional[str] = None
    umap_coordinates: List[float] = Field(default_factory=list)
    nearest_chunk_ids: List[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Response model for document data"""
    id: str
    title: str
    doc_type: str
    created_at: str
    chunk_count: int = 0


class VisualizationData(BaseModel):
    """Response model for visualization data"""
    documents: List[DocumentResponse]
    chunks: List[ChunkResponse]
    connections: List[Dict]


class DocumentCreateRequest(BaseModel):
    """Request model for document creation"""
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: str = Field(..., min_length=1, max_length=1000000, description="Document content")
    doc_type: str = Field(default="text", pattern="^(text|markdown|code|json)$", description="Document type")

    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()

    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class ChunkAttachmentRequest(BaseModel):
    """Request model for chunk attachment"""
    document_id: str = Field(..., min_length=1, max_length=100, description="Document ID to attach")

    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()