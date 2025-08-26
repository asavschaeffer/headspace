"""
Telegram Bot API Adapter

Converts Telegram webhook payloads into standardized InputMessage objects.
Handles text messages, photos, documents, voice messages, and other media types.

References:
- Telegram Bot API: https://core.telegram.org/bots/api
- Webhooks: https://core.telegram.org/bots/api#setwebhook
"""

import httpx
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..models import InputMessage, Attachment, AttachmentType

logger = logging.getLogger(__name__)


class TelegramAdapter:
    """Adapter for Telegram Bot API webhooks."""
    
    def __init__(self, bot_token: str, authorized_users: Optional[List[int]] = None):
        """
        Initialize the Telegram adapter.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            authorized_users: List of Telegram user IDs allowed to send thoughts
        """
        self.bot_token = bot_token
        self.authorized_users = set(authorized_users or [])
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def parse_webhook(self, payload: Dict[str, Any]) -> Optional[InputMessage]:
        """
        Parse a Telegram webhook payload into an InputMessage object.
        
        Args:
            payload: Raw webhook payload from Telegram
            
        Returns:
            InputMessage object or None if parsing fails
        """
        try:
            # Telegram webhook structure: update containing message/edited_message/etc.
            message = payload.get("message") or payload.get("edited_message")
            
            if not message:
                logger.debug("Telegram webhook contains no processable message")
                return None
            
            # Extract basic message info
            message_id = message.get("message_id")
            chat = message.get("chat", {})
            user = message.get("from", {})
            timestamp = message.get("date")
            
            # Check authorization if enabled
            user_id = user.get("id")
            if self.authorized_users and user_id not in self.authorized_users:
                logger.warning(f"Unauthorized Telegram user: {user_id}")
                return None
            
            # Convert timestamp
            try:
                msg_timestamp = datetime.fromtimestamp(timestamp)
            except (ValueError, TypeError):
                msg_timestamp = datetime.now()
            
            # Extract content based on message type
            content = None
            attachments = []
            
            # Text messages
            if "text" in message:
                content = message["text"].strip()
            
            # Photo messages
            elif "photo" in message:
                # Telegram sends multiple photo sizes, get the largest
                photos = message["photo"]
                largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                attachment = await self._download_telegram_file(largest_photo.get("file_id"), AttachmentType.IMAGE)
                if attachment:
                    attachments.append(attachment)
                
                # Photos can have captions
                caption = message.get("caption", "").strip()
                if caption:
                    content = caption
            
            # Document messages
            elif "document" in message:
                doc = message["document"]
                attachment = await self._download_telegram_file(doc.get("file_id"), AttachmentType.DOCUMENT, doc.get("file_name"))
                if attachment:
                    attachments.append(attachment)
                
                # Documents can have captions
                caption = message.get("caption", "").strip()
                if caption:
                    content = caption
            
            # Voice messages
            elif "voice" in message:
                voice = message["voice"]
                attachment = await self._download_telegram_file(voice.get("file_id"), AttachmentType.AUDIO)
                if attachment:
                    attachments.append(attachment)
            
            # Video messages
            elif "video" in message:
                video = message["video"]
                attachment = await self._download_telegram_file(video.get("file_id"), AttachmentType.VIDEO)
                if attachment:
                    attachments.append(attachment)
                
                # Videos can have captions
                caption = message.get("caption", "").strip()
                if caption:
                    content = caption
            
            # Audio files
            elif "audio" in message:
                audio = message["audio"]
                attachment = await self._download_telegram_file(audio.get("file_id"), AttachmentType.AUDIO, audio.get("file_name"))
                if attachment:
                    attachments.append(attachment)
            
            else:
                logger.info(f"Unsupported Telegram message type in message {message_id}")
                return None
            
            # Don't process messages with no content
            if not content and not attachments:
                logger.debug(f"Telegram message {message_id} has no processable content")
                return None
            
            # Build user identifier (prefer username, fall back to user ID)
            username = user.get("username")
            user_identifier = f"@{username}" if username else f"user_{user_id}"
            
            # Extract platform-specific metadata
            platform_metadata = {
                "telegram_message_id": message_id,
                "telegram_user_id": user_id,
                "telegram_username": username,
                "telegram_chat_id": chat.get("id"),
                "telegram_chat_type": chat.get("type"),
                "user_first_name": user.get("first_name"),
                "user_last_name": user.get("last_name"),
            }
            
            # Handle reply context
            reply_to = None
            reply_to_message = message.get("reply_to_message")
            if reply_to_message:
                reply_to = str(reply_to_message.get("message_id"))
                platform_metadata["reply_to_message"] = reply_to_message
            
            return InputMessage(
                content=content,
                attachments=attachments,
                source="telegram",
                user_identifier=user_identifier,
                timestamp=msg_timestamp,
                message_id=str(message_id),
                reply_to=reply_to,
                platform_metadata=platform_metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to parse Telegram webhook payload: {e}")
            return None
    
    async def _download_telegram_file(self, file_id: str, attachment_type: AttachmentType, filename: Optional[str] = None) -> Optional[Attachment]:
        """Download a file from Telegram and create an Attachment object."""
        
        if not file_id:
            logger.warning("Telegram file message missing file_id")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                # Step 1: Get file info from Telegram API
                file_info_response = await client.get(f"{self.base_url}/getFile?file_id={file_id}")
                file_info_response.raise_for_status()
                file_info = file_info_response.json()
                
                if not file_info.get("ok"):
                    logger.error(f"Telegram getFile API error: {file_info}")
                    return None
                
                file_path = file_info["result"].get("file_path")
                if not file_path:
                    logger.error(f"No file_path in Telegram file info: {file_info}")
                    return None
                
                # Step 2: Download the actual file content
                download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
                file_response = await client.get(download_url)
                file_response.raise_for_status()
                
                # Determine MIME type from file extension or default
                mime_type = "application/octet-stream"
                if filename:
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    elif filename.lower().endswith('.png'):
                        mime_type = "image/png"
                    elif filename.lower().endswith('.pdf'):
                        mime_type = "application/pdf"
                    elif filename.lower().endswith('.txt'):
                        mime_type = "text/plain"
                    # Add more as needed
                
                # For photos, try to get MIME type from response headers
                if attachment_type == AttachmentType.IMAGE:
                    content_type = file_response.headers.get("content-type")
                    if content_type and content_type.startswith("image/"):
                        mime_type = content_type
                
                return Attachment(
                    content=file_response.content,
                    mime_type=mime_type,
                    filename=filename,
                    attachment_type=attachment_type
                )
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to download Telegram file {file_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading Telegram file {file_id}: {e}")
            return None
    
    async def send_reply(self, chat_id: int, message: str, reply_to_message_id: Optional[int] = None) -> bool:
        """
        Send a reply message back to Telegram.
        
        Args:
            chat_id: Telegram chat ID to send to
            message: Text message to send
            reply_to_message_id: Optional ID of message to reply to
            
        Returns:
            True if message was sent successfully
        """
        try:
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"  # Allow basic formatting
            }
            
            # Add reply context if specified
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", json=payload)
                response.raise_for_status()
                
                result = response.json()
                if result.get("ok"):
                    logger.info(f"Telegram reply sent successfully to chat {chat_id}")
                    return True
                else:
                    logger.error(f"Telegram API error: {result}")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to send Telegram reply to chat {chat_id}: {e}")
            return False