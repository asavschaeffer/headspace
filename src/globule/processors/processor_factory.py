"""
ProcessorFactory for Phase 4 processor instantiation with dependency injection.

This factory creates processor instances with proper dependency injection,
following the Phase 3 patterns for provider instantiation and configuration.

Author: Globule Team
Version: 4.0.0 (Multi-Modal Extensions)
"""
import logging
from typing import List, Dict, Any, Optional

from globule.core.interfaces import IProcessor, IEmbeddingAdapter
from globule.config.settings import get_config
from .processor_router import ProcessorRouter
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    Factory for creating processor instances with dependency injection.
    
    Follows Phase 3 patterns for provider instantiation, ensuring proper
    dependency injection and configuration management.
    """
    
    def __init__(self, embedding_adapter: IEmbeddingAdapter):
        """
        Initialize the processor factory.
        
        Args:
            embedding_adapter: IEmbeddingAdapter instance for multimodal processors.
        """
        self.embedding_adapter = embedding_adapter
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def create_image_processor(self) -> ImageProcessor:
        """
        Create ImageProcessor with dependency injection.
        
        Returns:
            ImageProcessor instance with injected dependencies.
        """
        return ImageProcessor(self.embedding_adapter)
    
    def create_processor_router(self, processor_types: Optional[List[str]] = None) -> ProcessorRouter:
        """
        Create ProcessorRouter with registered processors.
        
        Args:
            processor_types: Optional list of processor types to register.
                           Defaults to ['image'] for Phase 4 scope.
        
        Returns:
            ProcessorRouter with processors registered.
        """
        if processor_types is None:
            processor_types = ['image']  # Phase 4 scope: image processing only
        
        router = ProcessorRouter()
        
        for processor_type in processor_types:
            try:
                processor = self._create_processor_by_type(processor_type)
                if processor:
                    router.register_processor(processor)
                    self.logger.debug(f"Registered {processor_type} processor")
            except Exception as e:
                self.logger.error(f"Failed to create {processor_type} processor: {e}")
        
        return router
    
    def _create_processor_by_type(self, processor_type: str) -> Optional[IProcessor]:
        """
        Create processor instance by type.
        
        Args:
            processor_type: Type of processor to create.
            
        Returns:
            IProcessor instance or None if type not supported.
        """
        if processor_type == 'image':
            return self.create_image_processor()
        else:
            self.logger.warning(f"Unknown processor type: {processor_type}")
            return None
    
    def get_available_processor_types(self) -> List[str]:
        """
        Get list of available processor types.
        
        Returns:
            List of supported processor type strings.
        """
        return ['image']  # Phase 4 scope: image processing only
    
    def create_all_processors(self) -> List[IProcessor]:
        """
        Create all available processor instances.
        
        Returns:
            List of all available IProcessor instances.
        """
        processors = []
        
        for processor_type in self.get_available_processor_types():
            try:
                processor = self._create_processor_by_type(processor_type)
                if processor:
                    processors.append(processor)
            except Exception as e:
                self.logger.error(f"Failed to create {processor_type} processor: {e}")
        
        return processors
    
    def get_factory_info(self) -> Dict[str, Any]:
        """
        Get factory configuration and status information.
        
        Returns:
            Dictionary with factory status and configuration.
        """
        return {
            "available_processors": self.get_available_processor_types(),
            "embedding_adapter": self.embedding_adapter.__class__.__name__,
            "factory_version": "4.0.0",
            "dependency_injection": True
        }