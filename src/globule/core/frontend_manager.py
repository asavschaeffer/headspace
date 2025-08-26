"""
Frontend Manager: Handles switching between different frontend implementations.

This module provides a unified interface for launching different frontends
(TUI, Web, CLI) while maintaining consistent functionality and state.
"""

import asyncio
import logging
import subprocess
import sys
import os
from typing import Optional, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class FrontendType(Enum):
    """Available frontend types."""
    TUI = "tui"
    WEB = "web"
    CLI = "cli"


class FrontendManager:
    """
    Manages different frontend implementations and provides switching capabilities.
    """
    
    def __init__(self):
        """Initialize the frontend manager."""
        self._frontends: Dict[FrontendType, Callable] = {}
        self._register_default_frontends()
    
    def _register_default_frontends(self):
        """Register default frontend implementations."""
        self._frontends[FrontendType.TUI] = self._launch_tui
        self._frontends[FrontendType.WEB] = self._launch_web
        self._frontends[FrontendType.CLI] = self._launch_cli
    
    def register_frontend(self, frontend_type: FrontendType, launcher: Callable):
        """
        Register a custom frontend implementation.
        
        Args:
            frontend_type: Type of frontend
            launcher: Function to launch the frontend
        """
        self._frontends[frontend_type] = launcher
        logger.info(f"Registered custom frontend: {frontend_type.value}")
    
    async def launch_frontend(self, frontend_type: FrontendType, **kwargs) -> Dict[str, Any]:
        """
        Launch the specified frontend.
        
        Args:
            frontend_type: Type of frontend to launch
            **kwargs: Arguments to pass to the frontend launcher
            
        Returns:
            Dict with launch result information
        """
        if frontend_type not in self._frontends:
            return {
                "success": False,
                "message": f"Frontend type {frontend_type.value} not registered",
                "data": None
            }
        
        try:
            logger.info(f"Launching {frontend_type.value} frontend...")
            launcher = self._frontends[frontend_type]
            result = await launcher(**kwargs)
            
            return {
                "success": True,
                "message": f"{frontend_type.value} frontend launched successfully",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Failed to launch {frontend_type.value} frontend: {e}")
            return {
                "success": False,
                "message": f"Failed to launch {frontend_type.value} frontend: {str(e)}",
                "data": {"error": str(e)}
            }
    
    async def _launch_tui(self, api: 'GlobuleAPI', topic: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Launch the clean TUI frontend in-process, passing the API instance."""
        try:
            from ..tui.clean_app import run_clean_tui

            # Run the CLEAN TUI with dependency injection - no god file!
            run_clean_tui(api, topic or "")

            return {
                "type": "tui",
                "topic": topic,
                "status": "closed"
            }

        except Exception as e:
            logger.error(f"TUI launch failed: {e}")
            # Re-raise to be caught by the main launch_frontend method
            raise Exception(f"TUI launch failed: {e}")
    
    async def _launch_web(self, port: int = 8000, host: str = "127.0.0.1", **kwargs) -> Dict[str, Any]:
        """Launch the web frontend."""
        try:
            # Check if we have the web frontend implementation
            try:
                from globule.interfaces.web.app import create_app
                
                logger.info(f"Starting web server on {host}:{port}")
                
                # Create FastAPI app
                app = create_app()
                
                # Start server (this would typically be done with uvicorn)
                import uvicorn
                
                # Run server in background
                config = uvicorn.Config(app, host=host, port=port, log_level="info")
                server = uvicorn.Server(config)
                
                # Start server in a separate task
                server_task = asyncio.create_task(server.serve())
                
                return {
                    "type": "web",
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "server_task": server_task,
                    "status": "launched"
                }
                
            except ImportError:
                # Web frontend not available, create placeholder
                logger.warning("Web frontend not implemented yet, creating placeholder")
                
                # Create a simple placeholder web server
                from http.server import HTTPServer, SimpleHTTPRequestHandler
                import threading
                
                class PlaceholderHandler(SimpleHTTPRequestHandler):
                    def do_GET(self):
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Globule Web Frontend</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 50px; }}
                                .container {{ max-width: 800px; margin: 0 auto; }}
                                .placeholder {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h1>Globule Web Frontend</h1>
                                <div class="placeholder">
                                    <h2>🚧 Coming Soon!</h2>
                                    <p>The web frontend is not yet implemented. This is a placeholder.</p>
                                    <p>Available features will include:</p>
                                    <ul>
                                        <li>Natural language search</li>
                                        <li>Draft management</li>
                                        <li>Canvas visualization</li>
                                        <li>Export functionality</li>
                                    </ul>
                                    <p><strong>For now, use:</strong></p>
                                    <ul>
                                        <li><code>globule draft</code> - Launch TUI</li>
                                        <li><code>globule search "query"</code> - CLI search</li>
                                    </ul>
                                </div>
                            </div>
                        </body>
                        </html>
                        """
                        
                        self.wfile.write(html_content.encode())
                
                server = HTTPServer((host, port), PlaceholderHandler)
                
                def run_server():
                    logger.info(f"Placeholder web server running on http://{host}:{port}")
                    server.serve_forever()
                
                # Start server in background thread
                server_thread = threading.Thread(target=run_server, daemon=True)
                server_thread.start()
                
                return {
                    "type": "web",
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "server": server,
                    "thread": server_thread,
                    "status": "placeholder_launched"
                }
                
        except Exception as e:
            raise Exception(f"Web launch failed: {e}")
    
    async def _launch_cli(self, **kwargs) -> Dict[str, Any]:
        """Launch CLI mode (essentially a no-op, as CLI is the default)."""
        return {
            "type": "cli",
            "message": "CLI mode is active (default)",
            "status": "active"
        }
    
    def list_available_frontends(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available frontend types with their capabilities.
        
        Returns:
            Dict mapping frontend types to their info
        """
        return {
            FrontendType.TUI.value: {
                "name": "Terminal User Interface",
                "description": "Interactive terminal-based interface with visual canvas",
                "capabilities": ["interactive_search", "visual_canvas", "real_time_updates"],
                "available": True
            },
            FrontendType.WEB.value: {
                "name": "Web Interface",
                "description": "Browser-based interface with responsive design",
                "capabilities": ["web_access", "responsive_design", "shareable_urls"],
                "available": True,
                "note": "Basic implementation with search and draft management"
            },
            FrontendType.CLI.value: {
                "name": "Command Line Interface",
                "description": "Scriptable command-line interface for automation",
                "capabilities": ["scriptable", "pipeable", "automation_friendly"],
                "available": True
            }
        }


# Global instance for easy access
frontend_manager = FrontendManager()