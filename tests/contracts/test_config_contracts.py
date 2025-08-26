"""
Contract compliance tests for configuration models and interfaces.

These tests ensure that the configuration contracts are well-defined
and can be implemented correctly.
"""
import pytest
from pydantic import ValidationError

from globule.config.models import EmbeddingConfig, StorageConfig, GlobuleConfig
from globule.config.errors import ConfigError, ConfigValidationError, ConfigFileError
from globule.core.interfaces import IConfigManager


class TestEmbeddingConfig:
    """Test EmbeddingConfig contract compliance."""
    
    def test_default_values(self):
        """Test that EmbeddingConfig has sensible defaults."""
        config = EmbeddingConfig()
        assert config.provider == "ollama"
        assert config.model == "mxbai-embed-large"
        assert config.endpoint is None
    
    def test_valid_providers(self):
        """Test that valid providers are accepted."""
        for provider in ["ollama", "huggingface", "openai"]:
            config = EmbeddingConfig(provider=provider)
            assert config.provider == provider
    
    def test_invalid_provider_rejected(self):
        """Test that unknown providers are rejected."""
        with pytest.raises(ValidationError, match="Input should be"):
            EmbeddingConfig(provider="unknown_provider")
    
    def test_https_endpoint_validation(self):
        """Test that only HTTPS endpoints are accepted."""
        # HTTPS should work
        config = EmbeddingConfig(endpoint="https://api.example.com")
        assert str(config.endpoint) == "https://api.example.com/"
        
        # HTTP should be rejected
        with pytest.raises(ValidationError, match="Embedding endpoint must use HTTPS"):
            EmbeddingConfig(endpoint="http://api.example.com")
    
    def test_none_endpoint_allowed(self):
        """Test that None endpoint is allowed (for defaults)."""
        config = EmbeddingConfig(endpoint=None)
        assert config.endpoint is None


class TestStorageConfig:
    """Test StorageConfig contract compliance."""
    
    def test_default_values(self):
        """Test that StorageConfig has sensible defaults."""
        config = StorageConfig()
        assert config.backend == "sqlite"
        assert config.path == ":memory:"
    
    def test_valid_backends(self):
        """Test that valid storage backends are accepted."""
        for backend in ["sqlite", "postgres"]:
            config = StorageConfig(backend=backend)
            assert config.backend == backend
    
    def test_invalid_backend_rejected(self):
        """Test that unknown backends are rejected."""
        with pytest.raises(ValidationError, match="Input should be"):
            StorageConfig(backend="unknown_backend")
    
    def test_custom_path(self):
        """Test that custom paths are accepted."""
        config = StorageConfig(path="/custom/path/database.db")
        assert config.path == "/custom/path/database.db"


class TestGlobuleConfig:
    """Test GlobuleConfig (root model) contract compliance."""
    
    def test_default_sections(self):
        """Test that GlobuleConfig includes all expected sections."""
        config = GlobuleConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.storage, StorageConfig)
    
    def test_nested_validation(self):
        """Test that nested validation works through the root model."""
        # Valid nested config
        config = GlobuleConfig(
            embedding={"provider": "huggingface", "model": "custom-model"},
            storage={"backend": "postgres", "path": "/data/db"}
        )
        assert config.embedding.provider == "huggingface"
        assert config.storage.backend == "postgres"
        
        # Invalid nested config should raise
        with pytest.raises(ValidationError):
            GlobuleConfig(embedding={"provider": "invalid_provider"})
    
    def test_partial_override(self):
        """Test that partial section overrides work correctly."""
        config = GlobuleConfig(embedding={"model": "custom-model"})
        # Should keep defaults for other fields
        assert config.embedding.provider == "ollama"  # default
        assert config.embedding.model == "custom-model"  # overridden


class TestConfigErrors:
    """Test configuration error hierarchy."""
    
    def test_config_error_base(self):
        """Test ConfigError base functionality."""
        error = ConfigError("Test message", source="test.yaml")
        assert str(error) == "Test message"
        assert error.source == "test.yaml"
    
    def test_config_validation_error(self):
        """Test ConfigValidationError inherits from ConfigError."""
        error = ConfigValidationError("Validation failed")
        assert isinstance(error, ConfigError)
    
    def test_config_file_error(self):
        """Test ConfigFileError inherits from ConfigError."""
        error = ConfigFileError("File not found")
        assert isinstance(error, ConfigError)


class DummyConfigManager(IConfigManager):
    """Dummy implementation for testing interface compliance."""
    
    def __init__(self):
        self._data = {"test": {"key": "value"}}
    
    def get(self, key: str, default=None):
        parts = key.split(".")
        value = self._data
        for part in parts:
            value = value.get(part, default)
            if value is default:
                break
        return value
    
    def get_section(self, section: str):
        return self._data.get(section, {})
    
    def reload(self):
        pass  # No-op for dummy


class TestIConfigManagerInterface:
    """Test IConfigManager interface compliance."""
    
    def test_dummy_implements_interface(self):
        """Test that dummy implementation correctly implements interface."""
        manager = DummyConfigManager()
        assert isinstance(manager, IConfigManager)
        
        # Test interface methods
        assert manager.get("test.key") == "value"
        assert manager.get("missing.key", "default") == "default"
        assert manager.get_section("test") == {"key": "value"}
        assert manager.get_section("missing") == {}
        
        # reload should not raise
        manager.reload()
    
    def test_interface_method_signatures(self):
        """Test that interface defines expected method signatures."""
        manager = DummyConfigManager()
        
        # get method should accept key and optional default
        result = manager.get("test.key")
        assert result == "value"
        
        result = manager.get("missing", "fallback")
        assert result == "fallback"
        
        # get_section should return dict
        section = manager.get_section("test")
        assert isinstance(section, dict)
        
        # reload should be callable
        manager.reload()  # Should not raise