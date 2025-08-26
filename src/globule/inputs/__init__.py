"""
Globule Input Sources Module

This module handles all external input sources (messaging platforms, email, etc.)
and converts them into standardized InputMessage objects for processing by the core engine.

The architecture is designed to be:
- Modular: Easy to add new input sources
- Standardized: All inputs become InputMessage objects
- Clean: Decoupled from core processing logic
"""

from .models import InputMessage, Attachment
from .manager import InputSourceManager

__all__ = ['InputMessage', 'Attachment', 'InputSourceManager']