"""
Configuration-related errors.
"""


class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    
    def __init__(self, message: str, source: str = None):
        super().__init__(message)
        self.source = source


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail validation."""
    pass


class ConfigFileError(ConfigError):
    """Raised when configuration files cannot be read or parsed."""
    pass