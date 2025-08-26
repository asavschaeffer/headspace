"""
Legacy configuration compatibility layer for Phase 3 migration.

Provides backward compatibility for existing code that uses the old
GlobuleConfig and get_config() patterns while transitioning to the
new PydanticConfigManager system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from globule.config.manager import PydanticConfigManager


@dataclass
class GlobuleConfig:
    """
    Legacy configuration class for backward compatibility.
    
    This class provides the same interface as the old configuration system
    but is backed by the new PydanticConfigManager.
    """
    
    def __init__(self, config_manager: Optional[PydanticConfigManager] = None):
        """Initialize with optional config manager."""
        self._config_manager = config_manager or PydanticConfigManager()
    
    @property
    def storage_path(self) -> str:
        """Storage path from new config system."""
        return self._config_manager.get('storage.path', ':memory:')
    
    @property 
    def default_embedding_model(self) -> str:
        """Default embedding model from new config system."""
        return self._config_manager.get('embedding.model', 'mxbai-embed-large')
    
    @property
    def default_parsing_model(self) -> str:
        """Default parsing model (legacy fallback)."""
        return "llama3.2:3b"  # Not yet in Phase 3 config
    
    @property
    def ollama_base_url(self) -> str:
        """Ollama base URL from new config system."""
        endpoint = self._config_manager.get('embedding.endpoint')
        if endpoint:
            return str(endpoint).rstrip('/')
        return "http://localhost:11434"
    
    @property
    def ollama_timeout(self) -> int:
        """Ollama timeout (legacy fallback)."""
        return 30  # Not yet in Phase 3 config
    
    @property
    def embedding_cache_size(self) -> int:
        """Embedding cache size (legacy fallback)."""
        return 1000  # Not yet in Phase 3 config
    
    @property
    def max_concurrent_requests(self) -> int:
        """Max concurrent requests (legacy fallback)."""
        return 5  # Not yet in Phase 3 config
    
    @property
    def default_schema(self) -> str:
        """Default schema (legacy fallback)."""
        return "default"  # Not yet in Phase 3 config
    
    @property
    def auto_schema_detection(self) -> bool:
        """Auto schema detection (legacy fallback)."""
        return True  # Not yet in Phase 3 config
    
    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the configuration file"""
        from globule.config.paths import user_config_path
        return user_config_path()
    
    @classmethod
    def load(cls) -> "GlobuleConfig":
        """Load configuration using new config system"""
        config_manager = PydanticConfigManager()
        return cls(config_manager)
    
    def save(self) -> None:
        """Save configuration (no-op for Phase 3)"""
        # In Phase 3, configuration is managed by the PydanticConfigManager
        # and saved through YAML files or environment variables
        pass
    
    def get_storage_dir(self) -> Path:
        """Get the storage directory as a Path object"""
        path = Path(self.storage_path)
        if path != Path(':memory:'):
            path.mkdir(exist_ok=True, parents=True)
        return path


# Global config instance
_config: Optional[GlobuleConfig] = None


def get_config() -> GlobuleConfig:
    """
    Get the global configuration instance using Phase 3 config system.
    
    This function provides backward compatibility while using the new
    PydanticConfigManager under the hood.
    """
    global _config
    if _config is None:
        _config = GlobuleConfig.load()
    return _config


def reload_config() -> GlobuleConfig:
    """Reload configuration from file"""
    global _config
    _config = GlobuleConfig.load()
    return _config