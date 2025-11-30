"""
Middleware configuration for Headspace System
Security headers, CORS, and compression middleware
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware


def setup_middleware(app: FastAPI):
    """Configure all middleware for the FastAPI application"""

    # Build allowed hosts dynamically for deployment environments
    allowed_hosts = [
        "localhost",
        "127.0.0.1",
        "*.localhost",
        "testserver",  # FastAPI TestClient
        "*.render.com",  # Render deployment
        "*.onrender.com",  # Render deployment (alternative domain)
    ]

    # Add any custom domains from environment
    custom_host = os.getenv("ALLOWED_HOST")
    if custom_host:
        allowed_hosts.append(custom_host)

    # Trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

    # GZip compression for responses
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS middleware for frontend integration
    cors_origins = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Same origin
        "http://127.0.0.1:8000",  # Localhost alternative
        "http://127.0.0.1:3000",  # React dev alternative
    ]

    # Add Render deployment origins
    render_app = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if render_app:
        cors_origins.extend([
            f"https://{render_app}",
            f"http://{render_app}",
        ])

    # Add any custom origin from environment
    custom_origin = os.getenv("CORS_ORIGIN")
    if custom_origin:
        cors_origins.append(custom_origin)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add security headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response