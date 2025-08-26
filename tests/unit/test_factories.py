"""
Tests for configuration-driven factories.

Tests the S5 factory system that uses PydanticConfigManager to
instantiate adapters and providers with proper dependency injection.
"""
import pytest
from unittest.mock import patch, MagicMock

from globule.config.manager import PydanticConfigManager
from globule.config.errors import ConfigError
from globule.core.factories import (
    EmbeddingAdapterFactory,
    StorageManagerFactory, 
    ParserProviderFactory,
    OrchestratorFactory,
    create_default_orchestrator
)
from globule.core.interfaces import IEmbeddingAdapter, IStorageManager, IParserProvider


class TestEmbeddingAdapterFactory:
    """Test embedding adapter factory with different providers."""
    
    def test_create_ollama_adapter(self):
        """Test creating Ollama embedding adapter."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'model': 'test-model',
                'endpoint': 'https://test:11434'  # Must be HTTPS for validation
            }
        })
        
        with patch('globule.services.embedding.ollama_provider.OllamaEmbeddingProvider') as mock_provider_cls, \
             patch('globule.services.embedding.ollama_adapter.OllamaEmbeddingAdapter') as mock_adapter_cls:
            
            mock_provider = MagicMock()
            mock_adapter = MagicMock()
            mock_provider_cls.return_value = mock_provider
            mock_adapter_cls.return_value = mock_adapter
            
            adapter = EmbeddingAdapterFactory.create(config)
            
            # Verify provider was created with correct config (URL gets trailing slash from Pydantic)
            mock_provider_cls.assert_called_once_with(
                base_url='https://test:11434/',
                model='test-model',
                timeout=30
            )
            
            # Verify adapter was created with provider
            mock_adapter_cls.assert_called_once_with(mock_provider)
            
            # Verify we got the adapter back
            assert adapter == mock_adapter
    
    def test_create_ollama_adapter_defaults(self):
        """Test creating Ollama adapter with default values."""
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'ollama'}
        })
        
        with patch('globule.services.embedding.ollama_provider.OllamaEmbeddingProvider') as mock_provider_cls, \
             patch('globule.services.embedding.ollama_adapter.OllamaEmbeddingAdapter') as mock_adapter_cls:
            
            EmbeddingAdapterFactory.create(config)
            
            # Verify provider created with defaults
            mock_provider_cls.assert_called_once_with(
                base_url=None,  # Will use provider default
                model='mxbai-embed-large',  # From config defaults
                timeout=30
            )
    
    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ConfigError."""
        # Create a mock config that returns an unsupported provider
        mock_config = MagicMock()
        mock_config.get.return_value = 'unsupported_provider'
        
        with pytest.raises(ConfigError, match="Unsupported embedding provider: unsupported_provider"):
            EmbeddingAdapterFactory.create(mock_config)
    
    def test_openai_provider_not_implemented(self):
        """Test that OpenAI provider raises not implemented error."""
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'openai'}
        })
        
        with pytest.raises(ConfigError, match="OpenAI embedding provider not yet implemented"):
            EmbeddingAdapterFactory.create(config)
    
    def test_huggingface_provider_not_implemented(self):
        """Test that HuggingFace provider raises not implemented error."""
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'huggingface'}
        })
        
        with pytest.raises(ConfigError, match="HuggingFace embedding provider not yet implemented"):
            EmbeddingAdapterFactory.create(config)


class TestStorageManagerFactory:
    """Test storage manager factory with different backends."""
    
    def test_create_sqlite_manager(self):
        """Test creating SQLite storage manager."""
        config = PydanticConfigManager(overrides={
            'storage': {
                'backend': 'sqlite',
                'path': '/test/db.sqlite'
            }
        })
        
        with patch('globule.storage.sqlite_adapter.SqliteStorageAdapter') as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            
            manager = StorageManagerFactory.create(config)
            
            # Verify adapter created with correct path
            mock_adapter_cls.assert_called_once_with(db_path='/test/db.sqlite')
            
            # Verify we got the adapter back
            assert manager == mock_adapter
    
    def test_create_sqlite_manager_defaults(self):
        """Test creating SQLite manager with default values."""
        config = PydanticConfigManager(overrides={
            'storage': {'backend': 'sqlite'}
        })
        
        with patch('globule.storage.sqlite_adapter.SqliteStorageAdapter') as mock_adapter_cls:
            StorageManagerFactory.create(config)
            
            # Verify adapter created with default path
            mock_adapter_cls.assert_called_once_with(db_path=':memory:')
    
    def test_unsupported_backend_raises_error(self):
        """Test that unsupported backend raises ConfigError."""
        # Create a mock config that returns an unsupported backend
        mock_config = MagicMock()
        mock_config.get.return_value = 'unsupported_backend'
        
        with pytest.raises(ConfigError, match="Unsupported storage backend: unsupported_backend"):
            StorageManagerFactory.create(mock_config)
    
    def test_postgres_backend_not_implemented(self):
        """Test that PostgreSQL backend raises not implemented error."""
        config = PydanticConfigManager(overrides={
            'storage': {'backend': 'postgres'}
        })
        
        with pytest.raises(ConfigError, match="PostgreSQL storage backend not yet implemented"):
            StorageManagerFactory.create(config)


class TestParserProviderFactory:
    """Test parser provider factory."""
    
    def test_create_mock_parser(self):
        """Test creating mock parser provider."""
        config = PydanticConfigManager()
        
        with patch('globule.services.providers_mock.MockParserProvider') as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser_cls.return_value = mock_parser
            
            parser = ParserProviderFactory.create(config)
            
            # Verify mock parser was created
            mock_parser_cls.assert_called_once()
            assert parser == mock_parser


class TestOrchestratorFactory:
    """Test orchestrator factory with full dependency injection."""
    
    def test_create_orchestrator_with_dependencies(self):
        """Test creating orchestrator with all dependencies."""
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'ollama', 'model': 'test-embed'},
            'storage': {'backend': 'sqlite', 'path': '/test/db'}
        })
        
        with patch('globule.core.factories.EmbeddingAdapterFactory.create') as mock_embed_factory, \
             patch('globule.core.factories.StorageManagerFactory.create') as mock_storage_factory, \
             patch('globule.core.factories.ParserProviderFactory.create') as mock_parser_factory, \
             patch('globule.orchestration.engine.GlobuleOrchestrator') as mock_orchestrator_cls:
            
            # Set up mocks
            mock_embedding = MagicMock()
            mock_storage = MagicMock()
            mock_parser = MagicMock()
            mock_orchestrator = MagicMock()
            
            mock_embed_factory.return_value = mock_embedding
            mock_storage_factory.return_value = mock_storage
            mock_parser_factory.return_value = mock_parser
            mock_orchestrator_cls.return_value = mock_orchestrator
            
            orchestrator = OrchestratorFactory.create(config)
            
            # Verify all factories were called with config
            mock_embed_factory.assert_called_once_with(config)
            mock_storage_factory.assert_called_once_with(config)
            mock_parser_factory.assert_called_once_with(config)
            
            # Verify orchestrator created with dependencies
            mock_orchestrator_cls.assert_called_once_with(
                parser_provider=mock_parser,
                embedding_provider=mock_embedding,
                storage_manager=mock_storage
            )
            
            assert orchestrator == mock_orchestrator


class TestCreateDefaultOrchestrator:
    """Test convenience function for creating default orchestrator."""
    
    def test_create_default_orchestrator_no_overrides(self):
        """Test creating orchestrator with default config."""
        with patch('globule.core.factories.PydanticConfigManager') as mock_config_cls, \
             patch('globule.core.factories.OrchestratorFactory.create') as mock_factory:
            
            mock_config = MagicMock()
            mock_orchestrator = MagicMock()
            mock_config_cls.return_value = mock_config
            mock_factory.return_value = mock_orchestrator
            
            orchestrator = create_default_orchestrator()
            
            # Verify config created without overrides
            mock_config_cls.assert_called_once_with(overrides=None)
            
            # Verify factory called with config
            mock_factory.assert_called_once_with(mock_config)
            
            assert orchestrator == mock_orchestrator
    
    def test_create_default_orchestrator_with_overrides(self):
        """Test creating orchestrator with config overrides."""
        overrides = {'embedding': {'provider': 'test'}}
        
        with patch('globule.core.factories.PydanticConfigManager') as mock_config_cls, \
             patch('globule.core.factories.OrchestratorFactory.create') as mock_factory:
            
            mock_config = MagicMock()
            mock_orchestrator = MagicMock()
            mock_config_cls.return_value = mock_config
            mock_factory.return_value = mock_orchestrator
            
            orchestrator = create_default_orchestrator(overrides)
            
            # Verify config created with overrides
            mock_config_cls.assert_called_once_with(overrides=overrides)
            
            # Verify factory called with config
            mock_factory.assert_called_once_with(mock_config)
            
            assert orchestrator == mock_orchestrator


class TestAdapterSelection:
    """Integration tests for adapter selection based on configuration."""
    
    def test_adapter_selection_by_provider(self):
        """Test that different providers result in different adapters."""
        # Test Ollama selection
        ollama_config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'ollama'}
        })
        
        with patch('globule.services.embedding.ollama_provider.OllamaEmbeddingProvider') as mock_provider, \
             patch('globule.services.embedding.ollama_adapter.OllamaEmbeddingAdapter') as mock_ollama:
            mock_provider.return_value = MagicMock()
            mock_ollama.return_value = MagicMock()
            EmbeddingAdapterFactory.create(ollama_config)
            mock_ollama.assert_called_once()
        
        # Test error for unsupported provider (using mock approach)
        bad_config = MagicMock()
        bad_config.get.return_value = 'nonexistent'
        
        with pytest.raises(ConfigError):
            EmbeddingAdapterFactory.create(bad_config)
    
    def test_storage_backend_selection(self):
        """Test that different backends result in different storage managers."""
        # Test SQLite selection
        sqlite_config = PydanticConfigManager(overrides={
            'storage': {'backend': 'sqlite'}
        })
        
        with patch('globule.storage.sqlite_adapter.SqliteStorageAdapter') as mock_sqlite:
            mock_sqlite.return_value = MagicMock()
            StorageManagerFactory.create(sqlite_config)
            mock_sqlite.assert_called_once()
        
        # Test error for unsupported backend (using mock approach)
        bad_config = MagicMock()
        bad_config.get.return_value = 'nonexistent'
        
        with pytest.raises(ConfigError):
            StorageManagerFactory.create(bad_config)
    
    def test_configuration_cascade_in_factories(self):
        """Test that configuration cascade works in factories."""
        # Test with environment override
        import os
        with patch.dict(os.environ, {'GLOBULE_EMBEDDING__PROVIDER': 'ollama'}, clear=False):
            with patch('globule.config.manager.system_config_path', return_value='/nonexistent'), \
                 patch('globule.config.manager.user_config_path', return_value='/nonexistent'), \
                 patch('globule.services.embedding.ollama_provider.OllamaEmbeddingProvider') as mock_provider, \
                 patch('globule.services.embedding.ollama_adapter.OllamaEmbeddingAdapter') as mock_adapter:
                
                mock_provider.return_value = MagicMock()
                mock_adapter.return_value = MagicMock()
                config = PydanticConfigManager()
                
                # Should use environment variable
                assert config.get('embedding.provider') == 'ollama'
                
                EmbeddingAdapterFactory.create(config)
                mock_adapter.assert_called_once()