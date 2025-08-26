"""
Configuration-driven factories for creating service providers.

Provides factory methods that use the Phase 3 configuration system
to instantiate embedding providers, storage managers, and other services
with proper dependency injection and adapter selection.
"""
from typing import Optional, Dict, Any

from globule.core.interfaces import IEmbeddingAdapter, IParserProvider, IStorageManager
from globule.config.manager import PydanticConfigManager
from globule.config.errors import ConfigError


class EmbeddingAdapterFactory:
    """Factory for creating embedding adapters based on configuration."""
    
    @staticmethod
    def create(config: PydanticConfigManager) -> IEmbeddingAdapter:
        """
        Create an embedding adapter based on configuration.
        
        Args:
            config: Configuration manager instance.
            
        Returns:
            Configured embedding adapter.
            
        Raises:
            ConfigError: If provider type is unsupported or configuration is invalid.
        """
        provider = config.get('embedding.provider')
        
        if provider == 'ollama':
            from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
            from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter
            
            # Create provider with configuration
            provider_instance = OllamaEmbeddingProvider(
                base_url=str(config.get('embedding.endpoint')) if config.get('embedding.endpoint') else None,
                model=config.get('embedding.model'),
                timeout=30  # Default timeout - could be configurable
            )
            
            # Wrap provider in adapter to match interface
            return OllamaEmbeddingAdapter(provider_instance)
            
        elif provider == 'openai':
            # Future: OpenAI adapter implementation
            raise ConfigError(f"OpenAI embedding provider not yet implemented")
            
        elif provider == 'huggingface':
            # Future: HuggingFace adapter implementation  
            raise ConfigError(f"HuggingFace embedding provider not yet implemented")
            
        else:
            raise ConfigError(f"Unsupported embedding provider: {provider}")


class StorageManagerFactory:
    """Factory for creating storage managers based on configuration."""
    
    @staticmethod
    def create(config: PydanticConfigManager) -> IStorageManager:
        """
        Create a storage manager based on configuration.
        
        Args:
            config: Configuration manager instance.
            
        Returns:
            Configured storage manager.
            
        Raises:
            ConfigError: If backend type is unsupported or configuration is invalid.
        """
        backend = config.get('storage.backend')
        
        if backend == 'sqlite':
            from globule.storage.sqlite_adapter import SqliteStorageAdapter
            
            # Create SQLite adapter with configuration
            return SqliteStorageAdapter(
                db_path=config.get('storage.path')
            )
            
        elif backend == 'postgres':
            # Future: PostgreSQL adapter implementation
            raise ConfigError(f"PostgreSQL storage backend not yet implemented")
            
        else:
            raise ConfigError(f"Unsupported storage backend: {backend}")


class ParserProviderFactory:
    """Factory for creating parser providers based on configuration."""
    
    @staticmethod  
    def create(config: PydanticConfigManager) -> IParserProvider:
        """
        Create a parser provider based on configuration.
        
        For now, return a mock parser as parsing configuration is not yet
        part of the Phase 3 config system.
        
        Args:
            config: Configuration manager instance.
            
        Returns:
            Configured parser provider.
        """
        # For Phase 3, use mock parser until parsing is properly configured
        from globule.services.providers_mock import MockParserProvider
        return MockParserProvider()


class OrchestratorFactory:
    """Factory for creating orchestrators with configured dependencies."""
    
    @staticmethod
    def create(config: PydanticConfigManager) -> 'GlobuleOrchestrator':
        """
        Create a GlobuleOrchestrator with all dependencies configured.
        
        Args:
            config: Configuration manager instance.
            
        Returns:
            Fully configured orchestrator instance.
        """
        from globule.orchestration.engine import GlobuleOrchestrator
        
        # Create all dependencies using factories
        embedding_adapter = EmbeddingAdapterFactory.create(config)
        storage_manager = StorageManagerFactory.create(config)
        parser_provider = ParserProviderFactory.create(config)
        
        return GlobuleOrchestrator(
            parser_provider=parser_provider,
            embedding_provider=embedding_adapter,
            storage_manager=storage_manager
        )


def create_default_orchestrator(overrides: Optional[Dict[str, Any]] = None) -> 'GlobuleOrchestrator':
    """
    Convenience function to create an orchestrator with default configuration.
    
    Args:
        overrides: Optional configuration overrides.
        
    Returns:
        Configured orchestrator instance.
    """
    config = PydanticConfigManager(overrides=overrides)
    return OrchestratorFactory.create(config)