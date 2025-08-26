"""
Configuration manager implementing three-tier cascade using BaseSettings.

Provides PydanticConfigManager that implements IConfigManager using
pydantic-settings for proper YAML and environment variable handling.
"""
import functools
from typing import Dict, Any, Optional, Tuple
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, InitSettingsSource
from pydantic_settings.sources import PydanticBaseSettingsSource

from globule.core.interfaces import IConfigManager
from .models import GlobuleConfig, EmbeddingConfig, StorageConfig
from .errors import ConfigError, ConfigValidationError
from .paths import system_config_path, user_config_path
from .sources import load_yaml_file, deep_merge


class MultiYamlSettingsSource(PydanticBaseSettingsSource):
    """
    Custom settings source for loading multiple YAML files in cascade order.
    
    Loads system YAML, then user YAML, and merges them properly.
    """
    
    def get_field_value(self, field_info, field_name: str) -> Tuple[Any, str, bool]:
        # Not used for nested data - only get_settings_data
        return None, field_name, False
    
    def prepare_field_value(self, field_name: str, field_value: Any, value_source: str) -> Any:
        return field_value
    
    def __call__(self) -> Dict[str, Any]:
        """Load and merge YAML configuration files."""
        system_config = load_yaml_file(system_config_path())
        user_config = load_yaml_file(user_config_path())
        
        # Merge system and user configs (user has higher precedence)
        merged_config = deep_merge(system_config, user_config)
        return merged_config


class PydanticConfigManager(BaseSettings, IConfigManager):
    """
    Configuration manager using pydantic-settings with cascade resolution.
    
    Implements full cascade: defaults → system → user → env → explicit overrides.
    Uses BaseSettings for proper type conversion and source handling.
    """
    
    # Configuration sections with proper type annotations
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    
    model_config = SettingsConfigDict(
        env_prefix='GLOBULE_',
        env_nested_delimiter='__',
        env_file_encoding='utf-8',
        case_sensitive=False,
        validate_default=True,
        extra='ignore'
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize configuration source priority.
        
        Order (highest to lowest precedence):
        1. Init settings (explicit overrides) - HIGHEST
        2. Environment variables  
        3. YAML files (system, then user) - LOWEST
        """
        return (
            init_settings,
            env_settings,
            MultiYamlSettingsSource(settings_cls),
        )
    
    def __init__(self, overrides: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize configuration manager.
        
        Args:
            overrides: Optional explicit overrides (highest precedence).
            **kwargs: Additional settings for BaseSettings.
            
        Raises:
            ConfigValidationError: If final configuration is invalid.
        """
        try:
            # Merge overrides into kwargs for init_settings
            if overrides:
                kwargs.update(overrides)
            
            super().__init__(**kwargs)
            
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to initialize configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'embedding.model').
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        try:
            return functools.reduce(getattr, key.split('.'), self)
        except AttributeError:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'embedding', 'storage').
            
        Returns:
            Dictionary containing all keys in the section.
        """
        try:
            section_model = getattr(self, section)
            if hasattr(section_model, 'model_dump'):
                return section_model.model_dump()
            return {}  # Or raise an error if section is not a model
        except AttributeError:
            return {}
    
    def reload(self) -> None:
        """
        Reload configuration from all sources.
        
        Note: For development use only. Production should use immutable configs.
        This creates a new instance and copies its values.
        """
        # Create new instance and copy values
        new_instance = self.__class__()
        for field_name in self.model_fields:
            setattr(self, field_name, getattr(new_instance, field_name))