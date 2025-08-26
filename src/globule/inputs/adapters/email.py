"""
Email Adapter

Converts email messages (via IMAP or webhook) into standardized InputMessage objects.
Handles text content, HTML content, and attachments.

This is a placeholder implementation - can be extended to support:
- IMAP polling (Gmail, Outlook, etc.)
- Email service webhooks (SendGrid, Mailgun, etc.)
- S3/SES processing
"""

import email
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ..models import InputMessage, Attachment, AttachmentType

logger = logging.getLogger(__name__)


class EmailAdapter:
    """Adapter for processing email messages."""
    
    def __init__(self, authorized_senders: Optional[List[str]] = None):
        """
        Initialize the email adapter.
        
        Args:
            authorized_senders: List of email addresses allowed to send thoughts
        """
        self.authorized_senders = set(authorized_senders or [])
    
    async def parse_email_message(self, raw_email: str, source_info: Optional[Dict[str, Any]] = None) -> Optional[InputMessage]:
        """
        Parse a raw email message into an InputMessage.
        
        Args:
            raw_email: Raw email content as string
            source_info: Additional source information (IMAP folder, etc.)
            
        Returns:
            InputMessage object or None if parsing fails
        """
        try:
            # Parse the email
            msg = email.message_from_string(raw_email)
            
            # Extract basic headers
            from_addr = msg.get("From", "").strip()
            subject = msg.get("Subject", "").strip()
            date_str = msg.get("Date", "")
            message_id = msg.get("Message-ID", "").strip()
            
            # Parse timestamp
            try:
                timestamp = email.utils.parsedate_to_datetime(date_str)
            except:
                timestamp = datetime.now()
            
            # Check authorization if enabled
            if self.authorized_senders and not any(sender in from_addr for sender in self.authorized_senders):
                logger.warning(f"Unauthorized email sender: {from_addr}")
                return None
            
            # Extract content and attachments
            content_parts = []
            attachments = []
            
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition", "")
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    # Plain text content
                    try:
                        text_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        content_parts.append(text_content.strip())
                    except:
                        logger.warning("Failed to decode email text content")
                
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    # HTML content (could convert to text)
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # TODO: Convert HTML to plain text for better processing
                        # For now, we'll skip HTML content
                        pass
                    except:
                        logger.warning("Failed to decode email HTML content")
                
                elif "attachment" in content_disposition or part.get_filename():
                    # File attachment
                    attachment = self._process_email_attachment(part)
                    if attachment:
                        attachments.append(attachment)
            
            # Combine content
            combined_content = "\n\n".join(content_parts).strip()
            
            # Include subject in content if we have one
            if subject:
                if combined_content:
                    combined_content = f"Subject: {subject}\n\n{combined_content}"
                else:
                    combined_content = f"Subject: {subject}"
            
            # Don't process empty emails
            if not combined_content and not attachments:
                logger.debug(f"Email {message_id} has no processable content")
                return None
            
            # Build platform metadata
            platform_metadata = {
                "email_message_id": message_id,
                "email_subject": subject,
                "email_from": from_addr,
                "source_info": source_info or {}
            }
            
            return InputMessage(
                content=combined_content if combined_content else None,
                attachments=attachments,
                source="email",
                user_identifier=from_addr,
                timestamp=timestamp,
                message_id=message_id,
                platform_metadata=platform_metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to parse email message: {e}")
            return None
    
    def _process_email_attachment(self, part) -> Optional[Attachment]:
        """Process an email attachment part into an Attachment object."""
        try:
            filename = part.get_filename()
            content_type = part.get_content_type()
            
            # Get the attachment data
            attachment_data = part.get_payload(decode=True)
            if not attachment_data:
                return None
            
            return Attachment(
                content=attachment_data,
                mime_type=content_type,
                filename=filename
            )
        
        except Exception as e:
            logger.error(f"Failed to process email attachment: {e}")
            return None


# TODO: Implement IMAP polling client
class IMAPEmailWatcher:
    """Watches an IMAP inbox for new emails and converts them to InputMessages."""
    
    def __init__(self, imap_server: str, username: str, password: str, folder: str = "INBOX"):
        self.imap_server = imap_server
        self.username = username
        self.password = password
        self.folder = folder
        self.adapter = EmailAdapter()
    
    async def poll_for_new_messages(self) -> List[InputMessage]:
        """Poll IMAP server for new messages. Returns list of InputMessages."""
        # TODO: Implement IMAP polling logic
        # This would use imaplib or aioimaplib to check for new emails
        # and convert them using the EmailAdapter
        return []