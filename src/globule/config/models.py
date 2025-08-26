"""
Pydantic configuration models for Globule.

These models define the structure and validation rules for configuration
sections, supporting the three-tier cascade system.
"""
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, Literal


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    
    provider: Literal["ollama", "huggingface", "openai"] = "ollama"
    model: str = "mxbai-embed-large"
    endpoint: Optional[HttpUrl] = None
    
    @field_validator("endpoint")
    @classmethod
    def require_https(cls, v):
        """Ensure embedding endpoints use HTTPS for security."""
        if v and not str(v).startswith("https"):
            raise ValueError("Embedding endpoint must use HTTPS")
        return v


class StorageConfig(BaseModel):
    """Configuration for storage backends."""
    
    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str = ":memory:"


class GlobuleConfig(BaseModel):
    """Root configuration model containing all sections."""
    
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()