"""
Middleware configuration for Headspace System
Security headers, CORS, and compression middleware
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware


def setup_middleware(app: FastAPI):
    """Configure all middleware for the FastAPI application"""

    # Trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "localhost",
            "127.0.0.1",
            "*.localhost",
            "testserver",
            "*.onrender.com",  # Allow all Render domains
            "asaschaeffer.com",  # Production domain
        ]
    )

    # GZip compression for responses
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React dev server
            "http://localhost:8000",  # Same origin
            "http://127.0.0.1:8000",  # Localhost alternative
            "http://127.0.0.1:3000",  # React dev alternative
        ],
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