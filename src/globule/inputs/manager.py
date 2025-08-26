"""
Input Source Manager

Coordinates all input sources and provides a unified interface for
registering, configuring, and processing messages from different platforms.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .models import InputMessage
from .adapters.whatsapp import WhatsAppAdapter
from .adapters.email import EmailAdapter
from .adapters.telegram import TelegramAdapter

logger = logging.getLogger(__name__)


class InputSourceManager:
    """
    Manages all input sources and provides a unified interface for processing messages.
    
    This class coordinates between different adapters and provides common functionality
    like authorization, rate limiting, and error handling.
    """
    
    def __init__(self):
        self.adapters: Dict[str, Any] = {}
        self.authorized_users: Dict[str, List[str]] = {}  # source -> list of authorized identifiers
        self.processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "last_processed": None
        }
    
    def register_whatsapp(self, access_token: str, verify_token: str, authorized_numbers: Optional[List[str]] = None) -> None:
        """
        Register WhatsApp as an input source.
        
        Args:
            access_token: WhatsApp Business API access token
            verify_token: Token for webhook verification
            authorized_numbers: List of phone numbers allowed to send messages
        """
        self.adapters["whatsapp"] = WhatsAppAdapter(access_token, verify_token)
        if authorized_numbers:
            self.authorized_users["whatsapp"] = authorized_numbers
        logger.info("WhatsApp input source registered")
    
    def register_telegram(self, bot_token: str, authorized_user_ids: Optional[List[int]] = None) -> None:
        """
        Register Telegram as an input source.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            authorized_user_ids: List of Telegram user IDs allowed to send messages
        """
        self.adapters["telegram"] = TelegramAdapter(bot_token, authorized_user_ids)
        if authorized_user_ids:
            self.authorized_users["telegram"] = [str(uid) for uid in authorized_user_ids]
        logger.info("Telegram input source registered")
    
    def register_email(self, authorized_senders: Optional[List[str]] = None) -> None:
        """
        Register Email as an input source.
        
        Args:
            authorized_senders: List of email addresses allowed to send messages
        """
        self.adapters["email"] = EmailAdapter(authorized_senders)
        if authorized_senders:
            self.authorized_users["email"] = authorized_senders
        logger.info("Email input source registered")
    
    async def process_whatsapp_webhook(self, payload: Dict[str, Any]) -> List[InputMessage]:
        """Process a WhatsApp webhook payload."""
        if "whatsapp" not in self.adapters:
            logger.error("WhatsApp adapter not registered")
            return []
        
        try:
            messages = await self.adapters["whatsapp"].parse_webhook(payload)
            self._update_stats(len(messages), 0)
            
            # Apply authorization filtering
            authorized_messages = []
            for msg in messages:
                if self._is_authorized(msg):
                    authorized_messages.append(msg)
                else:
                    logger.warning(f"Unauthorized message from {msg.user_identifier} via {msg.source}")
            
            return authorized_messages
        
        except Exception as e:
            logger.error(f"Failed to process WhatsApp webhook: {e}")
            self._update_stats(0, 1)
            return []
    
    async def process_telegram_webhook(self, payload: Dict[str, Any]) -> Optional[InputMessage]:
        """Process a Telegram webhook payload."""
        if "telegram" not in self.adapters:
            logger.error("Telegram adapter not registered")
            return None
        
        try:
            message = await self.adapters["telegram"].parse_webhook(payload)
            if message:
                if self._is_authorized(message):
                    self._update_stats(1, 0)
                    return message
                else:
                    logger.warning(f"Unauthorized message from {message.user_identifier} via {message.source}")
                    return None
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to process Telegram webhook: {e}")
            self._update_stats(0, 1)
            return None
    
    async def process_email_message(self, raw_email: str, source_info: Optional[Dict[str, Any]] = None) -> Optional[InputMessage]:
        """Process a raw email message."""
        if "email" not in self.adapters:
            logger.error("Email adapter not registered")
            return None
        
        try:
            message = await self.adapters["email"].parse_email_message(raw_email, source_info)
            if message:
                if self._is_authorized(message):
                    self._update_stats(1, 0)
                    return message
                else:
                    logger.warning(f"Unauthorized message from {message.user_identifier} via {message.source}")
                    return None
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to process email message: {e}")
            self._update_stats(0, 1)
            return None
    
    def verify_whatsapp_webhook(self, hub_mode: str, hub_verify_token: str, hub_challenge: str) -> Optional[str]:
        """Verify WhatsApp webhook during setup."""
        if "whatsapp" not in self.adapters:
            logger.error("WhatsApp adapter not registered")
            return None
        
        return self.adapters["whatsapp"].verify_webhook(hub_mode, hub_verify_token, hub_challenge)
    
    async def send_reply(self, source: str, user_identifier: str, message: str, reply_to_message_id: Optional[str] = None) -> bool:
        """
        Send a reply message back to the user via the appropriate platform.
        
        Args:
            source: The platform to send via (whatsapp, telegram, etc.)
            user_identifier: Platform-specific user identifier
            message: Message text to send
            reply_to_message_id: Optional message ID to reply to
            
        Returns:
            True if message was sent successfully
        """
        if source not in self.adapters:
            logger.error(f"Adapter for source '{source}' not registered")
            return False
        
        adapter = self.adapters[source]
        
        try:
            if source == "whatsapp" and hasattr(adapter, "send_reply"):
                return await adapter.send_reply(user_identifier, message, reply_to_message_id)
            elif source == "telegram" and hasattr(adapter, "send_reply"):
                # Convert user_identifier back to chat_id for Telegram
                chat_id = int(user_identifier.replace("@", "").replace("user_", ""))
                reply_id = int(reply_to_message_id) if reply_to_message_id else None
                return await adapter.send_reply(chat_id, message, reply_id)
            else:
                logger.warning(f"Reply not supported for source '{source}'")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send reply via {source}: {e}")
            return False
    
    def _is_authorized(self, message: InputMessage) -> bool:
        """Check if a message is from an authorized user."""
        source = message.source
        user_id = message.user_identifier
        
        # If no authorization list is configured for this source, allow all
        if source not in self.authorized_users:
            return True
        
        # Check if user is in the authorized list
        return user_id in self.authorized_users[source]
    
    def _update_stats(self, processed: int, failed: int) -> None:
        """Update processing statistics."""
        self.processing_stats["messages_processed"] += processed
        self.processing_stats["messages_failed"] += failed
        self.processing_stats["last_processed"] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.processing_stats,
            "active_sources": list(self.adapters.keys()),
            "authorized_users": {k: len(v) for k, v in self.authorized_users.items()}
        }
    
    def list_sources(self) -> List[str]:
        """Get list of registered input sources."""
        return list(self.adapters.keys())