"""
Minimal Cloud Relay Service for Globule Input Messages

This is a lightweight service that receives webhooks from messaging platforms
and forwards them to users' local Globule instances. It doesn't store any content,
just acts as a bridge between public webhooks and private local services.

Usage:
    python -m globule.inputs.relay_service --port 8080
"""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web, ClientSession
from aiohttp.web import Request, Response, json_response

logger = logging.getLogger(__name__)


@dataclass
class UserEndpoint:
    """Represents a user's local Globule endpoint."""
    user_id: str
    endpoint_url: str  # e.g., https://abc123.ngrok.io/webhook
    auth_token: str    # For authenticating messages to user
    platforms: List[str] = field(default_factory=list)  # whatsapp, telegram, etc.
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


class GlobuleRelayService:
    """
    Minimal relay service for forwarding webhook messages to local Globule instances.
    
    This service:
    1. Receives webhooks from messaging platforms
    2. Identifies the target user based on phone number/chat ID
    3. Forwards the message to the user's registered local endpoint
    4. Handles retries and basic error handling
    
    Privacy: No message content is stored, only routing information.
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.user_endpoints: Dict[str, UserEndpoint] = {}
        self.platform_users: Dict[str, Dict[str, str]] = {
            # Maps platform identifiers to user IDs
            # e.g., {"whatsapp": {"+1234567890": "user123"}}
        }
        self.stats = {
            "messages_relayed": 0,
            "messages_failed": 0,
            "users_registered": 0,
            "last_activity": None
        }
    
    def create_app(self) -> web.Application:
        """Create the aiohttp web application."""
        app = web.Application()
        
        # Health check
        app.router.add_get("/health", self.health_check)
        
        # User registration
        app.router.add_post("/register", self.register_user)
        app.router.add_delete("/register/{user_id}", self.unregister_user)
        
        # Platform webhooks
        app.router.add_post("/webhook/whatsapp", self.whatsapp_webhook)
        app.router.add_get("/webhook/whatsapp", self.whatsapp_verify)  # For verification
        app.router.add_post("/webhook/telegram/{bot_token}", self.telegram_webhook)
        
        # Admin endpoints
        app.router.add_get("/stats", self.get_stats)
        
        return app
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        return json_response({
            "status": "healthy",
            "service": "globule-relay",
            "users": len(self.user_endpoints),
            "uptime": "running"
        })
    
    async def register_user(self, request: Request) -> Response:
        """
        Register a user's local endpoint for message forwarding.
        
        POST /register
        {
            "user_id": "user123",
            "endpoint_url": "https://abc123.ngrok.io/webhook",
            "auth_token": "secret_token",
            "platforms": {
                "whatsapp": "+1234567890",
                "telegram": "user_123456"
            }
        }
        """
        try:
            data = await request.json()
            
            user_id = data.get("user_id")
            endpoint_url = data.get("endpoint_url")
            auth_token = data.get("auth_token")
            platforms_config = data.get("platforms", {})
            
            if not all([user_id, endpoint_url, auth_token]):
                return json_response(
                    {"error": "Missing required fields: user_id, endpoint_url, auth_token"},
                    status=400
                )
            
            # Create user endpoint
            user_endpoint = UserEndpoint(
                user_id=user_id,
                endpoint_url=endpoint_url,
                auth_token=auth_token,
                platforms=list(platforms_config.keys())
            )
            
            self.user_endpoints[user_id] = user_endpoint
            
            # Register platform mappings
            for platform, identifier in platforms_config.items():
                if platform not in self.platform_users:
                    self.platform_users[platform] = {}
                self.platform_users[platform][identifier] = user_id
            
            self.stats["users_registered"] += 1
            logger.info(f"Registered user {user_id} with platforms {list(platforms_config.keys())}")
            
            return json_response({
                "status": "registered",
                "user_id": user_id,
                "platforms": list(platforms_config.keys())
            })
        
        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            return json_response({"error": str(e)}, status=500)
    
    async def unregister_user(self, request: Request) -> Response:
        """Unregister a user."""
        user_id = request.match_info["user_id"]
        
        if user_id not in self.user_endpoints:
            return json_response({"error": "User not found"}, status=404)
        
        # Remove from platform mappings
        for platform_users in self.platform_users.values():
            to_remove = [k for k, v in platform_users.items() if v == user_id]
            for k in to_remove:
                del platform_users[k]
        
        del self.user_endpoints[user_id]
        logger.info(f"Unregistered user {user_id}")
        
        return json_response({"status": "unregistered", "user_id": user_id})
    
    async def whatsapp_verify(self, request: Request) -> Response:
        """Handle WhatsApp webhook verification."""
        hub_mode = request.query.get("hub.mode")
        hub_token = request.query.get("hub.verify_token") 
        hub_challenge = request.query.get("hub.challenge")
        
        # For simplicity, we'll accept any verification
        # In production, you'd want to verify the token
        if hub_mode == "subscribe":
            logger.info("WhatsApp webhook verified")
            return Response(text=hub_challenge)
        
        return Response(text="Verification failed", status=400)
    
    async def whatsapp_webhook(self, request: Request) -> Response:
        """Handle WhatsApp webhook messages."""
        try:
            payload = await request.json()
            
            # Extract phone number from WhatsApp payload
            phone_number = None
            try:
                entries = payload.get("entry", [])
                for entry in entries:
                    changes = entry.get("changes", [])
                    for change in changes:
                        if change.get("field") == "messages":
                            messages = change.get("value", {}).get("messages", [])
                            if messages:
                                phone_number = messages[0].get("from")
                                break
                    if phone_number:
                        break
            except Exception as e:
                logger.warning(f"Failed to extract phone number from WhatsApp payload: {e}")
            
            if not phone_number:
                return json_response({"error": "Could not identify phone number"}, status=400)
            
            # Find user for this phone number
            user_id = self.platform_users.get("whatsapp", {}).get(phone_number)
            if not user_id:
                logger.warning(f"No registered user for WhatsApp number {phone_number}")
                return json_response({"status": "no_user_registered"})
            
            # Forward to user's endpoint
            success = await self._forward_to_user(user_id, "whatsapp", payload)
            
            if success:
                self.stats["messages_relayed"] += 1
                return json_response({"status": "forwarded"})
            else:
                self.stats["messages_failed"] += 1
                return json_response({"error": "Forward failed"}, status=500)
        
        except Exception as e:
            logger.error(f"WhatsApp webhook error: {e}")
            return json_response({"error": str(e)}, status=500)
    
    async def telegram_webhook(self, request: Request) -> Response:
        """Handle Telegram webhook messages."""
        try:
            bot_token = request.match_info["bot_token"]
            payload = await request.json()
            
            # Extract user ID from Telegram payload
            user_identifier = None
            try:
                message = payload.get("message") or payload.get("edited_message")
                if message:
                    user = message.get("from", {})
                    username = user.get("username")
                    user_id = user.get("id")
                    user_identifier = f"@{username}" if username else f"user_{user_id}"
            except Exception as e:
                logger.warning(f"Failed to extract user from Telegram payload: {e}")
            
            if not user_identifier:
                return json_response({"error": "Could not identify user"}, status=400)
            
            # Find user for this Telegram identifier
            user_id = self.platform_users.get("telegram", {}).get(user_identifier)
            if not user_id:
                logger.warning(f"No registered user for Telegram identifier {user_identifier}")
                return json_response({"status": "no_user_registered"})
            
            # Forward to user's endpoint
            success = await self._forward_to_user(user_id, "telegram", payload)
            
            if success:
                self.stats["messages_relayed"] += 1
                return json_response({"status": "forwarded"})
            else:
                self.stats["messages_failed"] += 1
                return json_response({"error": "Forward failed"}, status=500)
        
        except Exception as e:
            logger.error(f"Telegram webhook error: {e}")
            return json_response({"error": str(e)}, status=500)
    
    async def _forward_to_user(self, user_id: str, platform: str, payload: Dict[str, Any]) -> bool:
        """Forward a message payload to a user's local endpoint."""
        user_endpoint = self.user_endpoints.get(user_id)
        if not user_endpoint:
            logger.error(f"User endpoint not found for {user_id}")
            return False
        
        try:
            # Create forwarding payload
            forward_payload = {
                "platform": platform,
                "payload": payload,
                "timestamp": datetime.now().isoformat(),
                "relay_metadata": {
                    "user_id": user_id,
                    "forwarded_from": "globule-relay"
                }
            }
            
            # Send to user's endpoint
            async with ClientSession() as session:
                async with session.post(
                    user_endpoint.endpoint_url,
                    json=forward_payload,
                    headers={
                        "Authorization": f"Bearer {user_endpoint.auth_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        user_endpoint.last_seen = datetime.now()
                        self.stats["last_activity"] = datetime.now()
                        logger.info(f"Successfully forwarded {platform} message to user {user_id}")
                        return True
                    else:
                        logger.error(f"User endpoint returned {response.status} for user {user_id}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to forward message to user {user_id}: {e}")
            return False
    
    async def get_stats(self, request: Request) -> Response:
        """Get service statistics."""
        return json_response({
            **self.stats,
            "active_users": len(self.user_endpoints),
            "platforms": list(self.platform_users.keys()),
            "last_activity": self.stats["last_activity"].isoformat() if self.stats["last_activity"] else None
        })


def create_relay_app(secret_key: str = "default-secret") -> web.Application:
    """Create and configure the relay service application."""
    relay = GlobuleRelayService(secret_key)
    return relay.create_app()


async def main():
    """Main entry point for running the relay service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Globule Message Relay Service")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--secret", default="globule-relay-secret", help="Secret key")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create app
    app = create_relay_app(args.secret)
    
    # Run server
    logger.info(f"Starting Globule Relay Service on {args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())