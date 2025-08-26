"""
Phase 4 Processor Extensions for Multi-Modal Support.

This package provides content-specific processors that extend the parsing
pipeline to handle multi-modal content types (images, audio, etc.) while
maintaining compatibility with the existing orchestration workflow.

Author: Globule Team
Version: 4.0.0 (Multi-Modal Extensions)
"""

from .processor_adapter import ProcessorAdapter
from .image_processor import ImageProcessor
from .processor_router import ProcessorRouter
from .processor_factory import ProcessorFactory

__all__ = ['ProcessorAdapter', 'ImageProcessor', 'ProcessorRouter', 'ProcessorFactory']