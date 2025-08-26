"""
Input Source Adapters

Each adapter converts a specific service's webhook/API format into our standardized InputMessage format.
This keeps the core engine completely decoupled from external service specifics.
"""

from .whatsapp import WhatsAppAdapter
from .email import EmailAdapter
from .telegram import TelegramAdapter

__all__ = ['WhatsAppAdapter', 'EmailAdapter', 'TelegramAdapter']