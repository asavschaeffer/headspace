"""
WhatsApp Business API Adapter

Converts WhatsApp webhook payloads into standardized InputMessage objects.
Handles text messages, images, documents, and other media types.

References:
- WhatsApp Cloud API: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks
- Message formats: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload
"""

import httpx
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..models import InputMessage, Attachment, AttachmentType

logger = logging.getLogger(__name__)


class WhatsAppAdapter:
    """Adapter for WhatsApp Business Cloud API webhooks."""
    
    def __init__(self, access_token: str, verify_token: str):
        """
        Initialize the WhatsApp adapter.
        
        Args:
            access_token: WhatsApp Business API access token
            verify_token: Token for webhook verification
        """
        self.access_token = access_token
        self.verify_token = verify_token
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def verify_webhook(self, hub_mode: str, hub_verify_token: str, hub_challenge: str) -> Optional[str]:
        """
        Verify WhatsApp webhook during setup.
        
        Args:
            hub_mode: Should be "subscribe"
            hub_verify_token: Should match our verify_token
            hub_challenge: Challenge string to echo back
            
        Returns:
            hub_challenge if verification succeeds, None otherwise
        """
        if hub_mode == "subscribe" and hub_verify_token == self.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return hub_challenge
        
        logger.warning(f"WhatsApp webhook verification failed: mode={hub_mode}, token_match={hub_verify_token == self.verify_token}")
        return None
    
    async def parse_webhook(self, payload: Dict[str, Any]) -> List[InputMessage]:
        """
        Parse a WhatsApp webhook payload into InputMessage objects.
        
        A single webhook can contain multiple messages, so we return a list.
        
        Args:
            payload: Raw webhook payload from WhatsApp
            
        Returns:
            List of InputMessage objects (one per message in the payload)
        """
        messages = []
        
        try:
            # WhatsApp webhook structure: entry[].changes[].value.messages[]
            entries = payload.get("entry", [])
            
            for entry in entries:
                changes = entry.get("changes", [])
                
                for change in changes:
                    value = change.get("value", {})
                    
                    # Skip if this isn't a message change
                    if change.get("field") != "messages":
                        continue
                    
                    # Process each message in this change
                    webhook_messages = value.get("messages", [])
                    for msg in webhook_messages:
                        try:
                            input_message = await self._parse_single_message(msg, value)
                            if input_message:
                                messages.append(input_message)
                        except Exception as e:
                            logger.error(f"Failed to parse WhatsApp message {msg.get('id', 'unknown')}: {e}")
                            # Continue processing other messages even if one fails
                            continue
        
        except Exception as e:
            logger.error(f"Failed to parse WhatsApp webhook payload: {e}")
            # Return empty list rather than crash
        
        return messages
    
    async def _parse_single_message(self, message: Dict[str, Any], webhook_value: Dict[str, Any]) -> Optional[InputMessage]:
        """Parse a single message from the webhook payload."""
        
        # Extract basic message info
        message_id = message.get("id")
        from_number = message.get("from")
        timestamp_str = message.get("timestamp")
        message_type = message.get("type")
        
        if not all([message_id, from_number, timestamp_str, message_type]):
            logger.warning(f"Incomplete WhatsApp message data: {message}")
            return None
        
        # Convert timestamp
        try:
            timestamp = datetime.fromtimestamp(int(timestamp_str))
        except (ValueError, TypeError):
            timestamp = datetime.now()
        
        # Extract content based on message type
        content = None
        attachments = []
        
        if message_type == "text":
            content = message.get("text", {}).get("body", "").strip()
        
        elif message_type == "image":
            attachment = await self._download_media_attachment(message.get("image", {}), AttachmentType.IMAGE)
            if attachment:
                attachments.append(attachment)
                # WhatsApp images can have captions
                caption = message.get("image", {}).get("caption", "").strip()
                if caption:
                    content = caption
        
        elif message_type == "document":
            attachment = await self._download_media_attachment(message.get("document", {}), AttachmentType.DOCUMENT)
            if attachment:
                attachments.append(attachment)
                # Documents can have captions
                caption = message.get("document", {}).get("caption", "").strip()
                if caption:
                    content = caption
        
        elif message_type == "audio":
            attachment = await self._download_media_attachment(message.get("audio", {}), AttachmentType.AUDIO)
            if attachment:
                attachments.append(attachment)
        
        elif message_type == "video":
            attachment = await self._download_media_attachment(message.get("video", {}), AttachmentType.VIDEO)
            if attachment:
                attachments.append(attachment)
                # Videos can have captions
                caption = message.get("video", {}).get("caption", "").strip()
                if caption:
                    content = caption
        
        else:
            logger.info(f"Unsupported WhatsApp message type: {message_type}")
            return None
        
        # Don't process messages with no content
        if not content and not attachments:
            logger.debug(f"WhatsApp message {message_id} has no processable content")
            return None
        
        # Extract platform-specific metadata
        platform_metadata = {
            "whatsapp_message_id": message_id,
            "whatsapp_message_type": message_type,
            "phone_number_id": webhook_value.get("metadata", {}).get("phone_number_id"),
            "display_phone_number": webhook_value.get("metadata", {}).get("display_phone_number"),
        }
        
        # Handle reply context
        reply_to = None
        context = message.get("context")
        if context:
            reply_to = context.get("id")  # ID of the message being replied to
            platform_metadata["reply_context"] = context
        
        return InputMessage(
            content=content,
            attachments=attachments,
            source="whatsapp",
            user_identifier=from_number,
            timestamp=timestamp,
            message_id=message_id,
            reply_to=reply_to,
            platform_metadata=platform_metadata
        )
    
    async def _download_media_attachment(self, media_data: Dict[str, Any], attachment_type: AttachmentType) -> Optional[Attachment]:
        """Download media from WhatsApp and create an Attachment object."""
        
        media_id = media_data.get("id")
        if not media_id:
            logger.warning("WhatsApp media message missing media ID")
            return None
        
        try:
            # Step 1: Get media URL from Facebook Graph API
            async with httpx.AsyncClient() as client:
                # Get media info and download URL
                media_url_response = await client.get(
                    f"{self.base_url}/{media_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                media_url_response.raise_for_status()
                media_info = media_url_response.json()
                
                download_url = media_info.get("url")
                if not download_url:
                    logger.error(f"No download URL in WhatsApp media info: {media_info}")
                    return None
                
                # Step 2: Download the actual media content
                media_response = await client.get(
                    download_url,
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                media_response.raise_for_status()
                
                # Extract metadata
                content_type = media_response.headers.get("content-type", "application/octet-stream")
                filename = media_data.get("filename")  # Available for documents
                
                return Attachment(
                    content=media_response.content,
                    mime_type=content_type,
                    filename=filename,
                    attachment_type=attachment_type
                )
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to download WhatsApp media {media_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading WhatsApp media {media_id}: {e}")
            return None
    
    async def send_reply(self, to_number: str, message: str, reply_to_message_id: Optional[str] = None) -> bool:
        """
        Send a reply message back to WhatsApp.
        
        Args:
            to_number: Phone number to send to (same format as received)
            message: Text message to send
            reply_to_message_id: Optional ID of message to reply to
            
        Returns:
            True if message was sent successfully
        """
        try:
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "text",
                "text": {"body": message}
            }
            
            # Add reply context if specified
            if reply_to_message_id:
                payload["context"] = {"message_id": reply_to_message_id}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"WhatsApp reply sent successfully to {to_number}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to send WhatsApp reply to {to_number}: {e}")
            return False