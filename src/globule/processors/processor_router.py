"""
ProcessorRouter for Phase 4 multi-modal processing coordination.

This router coordinates between different processor types, selecting the best
processor for each content type and managing the processing workflow.

Author: Globule Team
Version: 4.0.0 (Multi-Modal Extensions)
"""
import logging
from typing import List, Optional, Dict, Any
from collections import defaultdict

from globule.core.interfaces import IProcessor
from globule.core.models import GlobuleV1, ProcessedContent
from globule.core.errors import ParserError

logger = logging.getLogger(__name__)


class ProcessorRouter:
    """
    Router that coordinates content processing across multiple processor types.
    
    The router evaluates confidence scores from different processors and
    selects the best match for each content type, enabling extensible
    multi-modal processing capabilities.
    """
    
    def __init__(self):
        """Initialize the processor router."""
        self._processors: List[IProcessor] = []
        self.logger = logging.getLogger(__name__)
    
    def register_processor(self, processor: IProcessor) -> None:
        """
        Register a processor with the router.
        
        Args:
            processor: IProcessor implementation to register.
        """
        self._processors.append(processor)
        self.logger.debug(f"Registered processor: {processor.get_processor_type()}")
    
    def get_registered_processors(self) -> List[IProcessor]:
        """Get list of all registered processors."""
        return self._processors.copy()
    
    async def route_and_process(self, globule: GlobuleV1) -> ProcessedContent:
        """
        Route content to the best processor and process it.
        
        Evaluates confidence scores from all registered processors and
        selects the one with the highest confidence to process the content.
        
        Args:
            globule: The globule to process.
            
        Returns:
            ProcessedContent from the selected processor.
            
        Raises:
            ParserError: If no processor can handle the content or processing fails.
        """
        if not self._processors:
            raise ParserError("No processors registered with router")
        
        # Evaluate confidence scores from all processors
        confidence_scores = {}
        for processor in self._processors:
            try:
                confidence = processor.can_process(globule)
                confidence_scores[processor] = confidence
                self.logger.debug(
                    f"Processor {processor.get_processor_type()} confidence: {confidence:.3f}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error evaluating processor {processor.get_processor_type()}: {e}"
                )
                confidence_scores[processor] = 0.0
        
        # Select best processor
        best_processor = max(confidence_scores.items(), key=lambda x: x[1])
        processor, confidence = best_processor
        
        if confidence == 0.0:
            raise ParserError(f"No processor can handle content: {globule.raw_text[:50]}...")
        
        self.logger.info(
            f"Selected processor {processor.get_processor_type()} "
            f"with confidence {confidence:.3f}"
        )
        
        # Process with selected processor
        try:
            return await processor.process(globule)
        except Exception as e:
            self.logger.error(
                f"Processing failed with {processor.get_processor_type()}: {e}"
            )
            raise ParserError(f"Processing failed: {e}") from e
    
    def get_processor_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities summary for all registered processors.
        
        Returns:
            Dictionary mapping processor types to their supported capabilities.
        """
        capabilities = defaultdict(list)
        
        for processor in self._processors:
            processor_type = processor.get_processor_type()
            
            # Add basic capability info
            capabilities[processor_type].append("content_processing")
            
            # Add specific capabilities based on processor type
            if processor_type == "image":
                capabilities[processor_type].extend([
                    "file_path_detection", 
                    "url_detection", 
                    "base64_detection",
                    "exif_extraction",
                    "thumbnail_generation"
                ])
            elif processor_type == "text":
                capabilities[processor_type].extend([
                    "nlp_parsing",
                    "content_structuring"
                ])
        
        return dict(capabilities)
    
    async def process_batch(self, globules: List[GlobuleV1]) -> List[ProcessedContent]:
        """
        Process multiple globules efficiently.
        
        Routes each globule to the appropriate processor and processes them.
        Provides error isolation - individual failures don't stop the batch.
        
        Args:
            globules: List of globules to process.
            
        Returns:
            List of ProcessedContent results. Failed items will have error details
            in their metadata.
        """
        results = []
        
        for i, globule in enumerate(globules):
            try:
                result = await self.route_and_process(globule)
                results.append(result)
                self.logger.debug(f"Batch item {i+1}/{len(globules)} processed successfully")
            except Exception as e:
                # Create fallback result for failed items
                self.logger.error(f"Batch item {i+1}/{len(globules)} failed: {e}")
                
                fallback_result = ProcessedContent(
                    structured_data={
                        "title": globule.raw_text[:50] + "..." if len(globule.raw_text) > 50 else globule.raw_text,
                        "category": "error",
                        "domain": "general",
                        "content_type": "unknown",
                        "error": str(e)
                    },
                    metadata={
                        "batch_error": True,
                        "error_details": str(e),
                        "globule_id": str(globule.globule_id),
                        "processor_error": True
                    },
                    confidence=0.0,
                    processor_type="fallback",
                    processing_time_ms=0.0
                )
                results.append(fallback_result)
        
        self.logger.info(f"Batch processing completed: {len(results)} items processed")
        return results
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processor routing.
        
        Returns:
            Dictionary with routing statistics and processor health info.
        """
        return {
            "registered_processors": len(self._processors),
            "processor_types": [p.get_processor_type() for p in self._processors],
            "capabilities": self.get_processor_capabilities(),
            "router_version": "4.0.0"
        }