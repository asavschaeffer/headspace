from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class Document(BaseModel):
    id: str
    title: str
    content: str
    doc_type: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    chunk_type: str
    embedding: List[float] = Field(default_factory=list)
    position_3d: List[float] = Field(default_factory=list)
    color: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[str] = Field(default_factory=list)

    # Enhanced Metadata Fields
    tags: List[str] = Field(default_factory=list)
    tag_confidence: Dict[str, float] = Field(default_factory=dict)
    texture: str = "smooth"
    shape_3d: Any = Field(default_factory=lambda: {"type": "sphere"})
    reasoning: str = ""
    embedding_model: str = ""
    timestamp_created: datetime = Field(default_factory=_now_utc)
    timestamp_modified: datetime = Field(default_factory=_now_utc)
    cluster_id: Optional[int] = None
    cluster_confidence: Optional[float] = None
    cluster_label: Optional[str] = None
    umap_coordinates: List[float] = Field(default_factory=list)
    nearest_chunk_ids: List[str] = Field(default_factory=list)

class ChunkConnection(BaseModel):
    from_chunk_id: str
    to_chunk_id: str
    connection_type: str
    strength: float
