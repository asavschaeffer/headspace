"""
Defines the application's custom exception hierarchy.

This ensures that the core engine handles a consistent set of errors,
regardless of the specific provider (parser, storage, etc.) being used.
Adapters are responsible for translating provider-specific exceptions into
these core exception types at the boundary.
"""

class GlobuleError(Exception):
    """Base exception class for all application-specific errors."""
    pass

class ContractsError(GlobuleError):
    """Errors related to data or interface contract violations."""
    pass

class OrchestrationError(GlobuleError):
    """Errors arising from the main orchestration engine."""
    pass

class ParserError(GlobuleError):
    """Errors related to parsing content."""
    pass

class EmbeddingError(GlobuleError):
    """Errors related to generating embeddings."""
    pass

class StorageError(GlobuleError):
    """Errors related to the storage backend."""
    pass
