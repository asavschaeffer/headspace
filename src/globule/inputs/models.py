"""
Standardized Input Models

These models provide a unified interface for all external input sources.
Every adapter (WhatsApp, Telegram, Email, etc.) converts its specific format
into these standardized objects for processing by the core engine.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AttachmentType(str, Enum):
    """Types of attachments we can process."""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class Attachment(BaseModel):
    """Represents a file (image, doc, etc.) attached to a message."""
    content: bytes
    mime_type: str
    filename: Optional[str] = None
    attachment_type: AttachmentType = AttachmentType.UNKNOWN
    size_bytes: Optional[int] = None
    
    class Config:
        # Allow arbitrary types for bytes content
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-detect attachment type from mime_type
        if not hasattr(self, 'attachment_type') or self.attachment_type == AttachmentType.UNKNOWN:
            self.attachment_type = self._detect_type_from_mime()
            
        # Auto-calculate size if not provided
        if self.size_bytes is None:
            self.size_bytes = len(self.content)
    
    def _detect_type_from_mime(self) -> AttachmentType:
        """Auto-detect attachment type from MIME type."""
        mime_lower = self.mime_type.lower()
        
        if mime_lower.startswith('image/'):
            return AttachmentType.IMAGE
        elif mime_lower.startswith('video/'):
            return AttachmentType.VIDEO
        elif mime_lower.startswith('audio/'):
            return AttachmentType.AUDIO
        elif mime_lower in ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return AttachmentType.DOCUMENT
        else:
            return AttachmentType.UNKNOWN


class InputMessage(BaseModel):
    """
    A standardized message object that represents any input from external sources.
    
    This is the universal format that all adapters convert their specific
    webhook/API formats into, allowing the core engine to process any input source uniformly.
    """
    
    # Core content
    content: Optional[str] = None  # The text of the message
    attachments: List[Attachment] = Field(default_factory=list)
    
    # Source identification
    source: str  # e.g., "whatsapp", "telegram", "email", "cli"
    user_identifier: str  # e.g., phone number, email address, telegram_chat_id
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: Optional[str] = None  # Original message ID from the source platform
    thread_id: Optional[str] = None  # For threaded conversations
    reply_to: Optional[str] = None  # If this is a reply to another message
    
    # Platform-specific metadata (for features like auto-reply)
    platform_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing hints
    priority: int = Field(default=5, ge=1, le=10)  # 1=lowest, 10=highest priority
    auto_process: bool = True  # Whether to automatically process this message
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: f"<{len(v)} bytes>"  # Don't actually serialize bytes content
        }
    
    @property
    def has_content(self) -> bool:
        """Check if this message has any processable content."""
        return bool(self.content or self.attachments)
    
    @property
    def content_summary(self) -> str:
        """Generate a brief summary of this message's content."""
        parts = []
        
        if self.content:
            preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
            parts.append(f"Text: {preview}")
        
        if self.attachments:
            attachment_summary = ", ".join([f"{att.attachment_type.value}" for att in self.attachments])
            parts.append(f"Attachments: {attachment_summary}")
        
        return " | ".join(parts) if parts else "Empty message"
    
    def to_globule_source_metadata(self) -> Dict[str, Any]:
        """
        Convert this InputMessage into metadata suitable for storing with processed globules.
        This preserves the context of where the content came from.
        """
        return {
            "input_source": self.source,
            "user_identifier": self.user_identifier,
            "original_timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "reply_to": self.reply_to,
            "attachment_count": len(self.attachments),
            "attachment_types": [att.attachment_type.value for att in self.attachments],
            "platform_metadata": self.platform_metadata
        }