"""
Base ProcessorAdapter for the Phase 4 processor extension system.

This module provides the foundation for content-specific processors that extend
the parsing pipeline to handle multi-modal content (text, images, audio, etc.)
following the established adapter pattern from Phase 2.

Author: Globule Team  
Version: 4.0.0 (Multi-Modal Extensions)
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from globule.core.interfaces import IProcessor
from globule.core.models import GlobuleV1, ProcessedContent
from globule.core.errors import ParserError
from globule.config.settings import get_config

logger = logging.getLogger(__name__)


class ProcessorAdapter(IProcessor):
    """
    Base adapter class for content processors.
    
    Provides common functionality for all processor types including:
    - Error handling and recovery
    - Performance timing
    - Configuration access
    - Logging integration
    
    Follows the Phase 2 adapter pattern for consistent error boundaries
    and provider abstraction.
    """
    
    def __init__(self, processor_type: str):
        """
        Initialize the processor adapter.
        
        Args:
            processor_type: String identifier for this processor type
        """
        self.processor_type = processor_type
        self.config = get_config()
        self.logger = logging.getLogger(f"{__name__}.{processor_type}")
    
    def can_process(self, globule: GlobuleV1) -> float:
        """
        Determine processing capability for given content.
        
        Base implementation provides content profiling similar to
        AdaptiveInputModule patterns. Subclasses should override
        for specific content type detection.
        
        Args:
            globule: The raw globule to evaluate.
            
        Returns:
            Confidence score 0.0-1.0 indicating processing capability.
        """
        try:
            # Basic content checks
            if not globule.raw_text or not globule.raw_text.strip():
                return 0.0
            
            # Delegate to specific processor logic
            return self._calculate_processing_confidence(globule)
            
        except Exception as e:
            self.logger.warning(f"Error calculating processing confidence: {e}")
            return 0.0
    
    async def process(self, globule: GlobuleV1) -> ProcessedContent:
        """
        Process globule content with error handling and timing.
        
        Provides consistent error boundaries and performance monitoring
        across all processor implementations.
        
        Args:
            globule: The globule to process.
            
        Returns:
            ProcessedContent with extracted structured data and metadata.
            
        Raises:
            ParserError: If processing fails in a recoverable way.
        """
        start_time = time.time()
        
        try:
            # Pre-processing validation
            self._validate_input(globule)
            
            # Delegate to specific processor implementation
            structured_data, metadata = await self._process_content(globule)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Add adapter metadata
            adapter_metadata = {
                "processed_at": time.time(),
                "globule_id": str(globule.globule_id),
                "adapter_version": "4.0.0",
                **metadata
            }
            
            # Determine confidence from structured data or use default
            confidence = structured_data.get('confidence_score', 0.8)
            if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                # Remove from structured data to avoid duplication
                structured_data.pop('confidence_score', None)
            else:
                confidence = 0.8
            
            return ProcessedContent(
                structured_data=structured_data,
                metadata=adapter_metadata,
                confidence=confidence,
                processor_type=self.processor_type,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Processing failed for {self.processor_type}: {e}")
            
            # Create fallback result for graceful degradation
            return self._create_fallback_result(globule, str(e), processing_time_ms)
    
    def get_processor_type(self) -> str:
        """Return the processor type identifier."""
        return self.processor_type
    
    @abstractmethod
    def _calculate_processing_confidence(self, globule: GlobuleV1) -> float:
        """
        Calculate specific processing confidence for this processor type.
        
        Args:
            globule: The globule to evaluate.
            
        Returns:
            Confidence score 0.0-1.0.
        """
        pass
    
    @abstractmethod
    async def _process_content(self, globule: GlobuleV1) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process the specific content type.
        
        Args:
            globule: The globule to process.
            
        Returns:
            Tuple of (structured_data, metadata) dictionaries.
            
        Raises:
            Exception: If processing fails.
        """
        pass
    
    def _validate_input(self, globule: GlobuleV1) -> None:
        """
        Validate input globule for processing.
        
        Args:
            globule: The globule to validate.
            
        Raises:
            ParserError: If validation fails.
        """
        if not isinstance(globule, GlobuleV1):
            raise ParserError(f"Invalid globule type: {type(globule)}")
        
        if not globule.raw_text:
            raise ParserError("Empty or missing raw_text in globule")
    
    def _create_fallback_result(self, globule: Any, error_message: str, processing_time_ms: float) -> ProcessedContent:
        """
        Create fallback result when processing fails.
        
        Args:
            globule: The original globule.
            error_message: Description of the error.
            processing_time_ms: Time spent processing.
            
        Returns:
            ProcessedContent with minimal fallback data.
        """
        # Safely extract text and ID from potentially invalid globule
        try:
            if hasattr(globule, 'raw_text') and globule.raw_text:
                text = globule.raw_text[:50] + "..." if len(globule.raw_text) > 50 else globule.raw_text
                globule_id = str(globule.globule_id) if hasattr(globule, 'globule_id') else "unknown"
            else:
                text = "Invalid input"
                globule_id = "unknown"
        except Exception:
            text = "Processing error"
            globule_id = "unknown"
        
        fallback_data = {
            "title": text,
            "category": "note",
            "domain": "general", 
            "content_type": "unknown",
            "error": error_message
        }
        
        fallback_metadata = {
            "processor_error": True,
            "error_details": error_message,
            "globule_id": globule_id,
            "fallback_result": True
        }
        
        return ProcessedContent(
            structured_data=fallback_data,
            metadata=fallback_metadata,
            confidence=0.1,  # Low confidence for fallback
            processor_type=self.processor_type,
            processing_time_ms=processing_time_ms
        )
    
    def _extract_content_profile(self, text: str) -> Dict[str, Any]:
        """
        Extract content profile similar to AdaptiveInputModule patterns.
        
        Args:
            text: Input text to profile.
            
        Returns:
            Dictionary with content characteristics.
        """
        if not text:
            return {}
        
        profile = {
            "length": len(text),
            "word_count": len(text.split()),
            "line_count": text.count('\n') + 1,
            "has_urls": 'http' in text.lower(),
            "has_code": '```' in text or 'def ' in text or 'function ' in text,
            "is_question": text.strip().endswith('?'),
            "estimated_tokens": len(text.split()) * 1.3
        }
        
        # Calculate complexity scores
        profile["structure_score"] = min(1.0, profile["line_count"] / max(profile["word_count"], 1) * 10)
        profile["technical_score"] = 0.8 if profile["has_code"] else 0.1 if profile["has_urls"] else 0.0
        profile["creativity_score"] = min(1.0, len(set(text.lower().split())) / max(profile["word_count"], 1) * 2)
        
        return profile