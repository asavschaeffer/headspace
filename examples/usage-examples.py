#!/usr/bin/env python3
"""
Globule Configuration Usage Examples

This file demonstrates various ways to configure and use Globule
with the Phase 3 configuration system.
"""

import os
from pathlib import Path
from globule.config.manager import PydanticConfigManager
from globule.core.factories import create_default_orchestrator, EmbeddingAdapterFactory, StorageManagerFactory

def example_basic_usage():
    """Basic configuration usage with defaults."""
    print("=== Basic Usage ===")
    
    # Use default configuration (ollama + sqlite)
    config = PydanticConfigManager()
    print(f"Default provider: {config.get('embedding.provider')}")
    print(f"Default model: {config.get('embedding.model')}")
    print(f"Default storage: {config.get('storage.backend')}")
    
    # Note: Orchestrator creation requires actual services
    print("Orchestrator creation requires running Ollama/SQLite services")
    print("Use create_default_orchestrator() when services are available")

def example_explicit_overrides():
    """Configuration with explicit overrides."""
    print("\n=== Explicit Overrides ===")
    
    # Override specific settings
    config_overrides = {
        'embedding': {
            'provider': 'ollama',
            'model': 'llama3.2:3b',
            'endpoint': 'https://localhost:11434'
        },
        'storage': {
            'backend': 'sqlite',
            'path': './custom-database.db'
        }
    }
    
    config = PydanticConfigManager(overrides=config_overrides)
    print(f"Override provider: {config.get('embedding.provider')}")
    print(f"Override model: {config.get('embedding.model')}")
    print(f"Override storage path: {config.get('storage.path')}")
    
    # Note: Orchestrator creation requires actual services  
    print("Use create_default_orchestrator(config_overrides) when services are available")

def example_environment_variables():
    """Configuration using environment variables."""
    print("\n=== Environment Variables ===")
    
    # Set environment variables (highest precedence)
    os.environ['GLOBULE_EMBEDDING__PROVIDER'] = 'ollama'
    os.environ['GLOBULE_EMBEDDING__MODEL'] = 'codellama:7b'
    os.environ['GLOBULE_STORAGE__BACKEND'] = 'sqlite'
    os.environ['GLOBULE_STORAGE__PATH'] = '/tmp/env-globule.db'
    
    config = PydanticConfigManager()
    print(f"Env provider: {config.get('embedding.provider')}")
    print(f"Env model: {config.get('embedding.model')}")
    print(f"Env storage path: {config.get('storage.path')}")
    
    # Clean up environment
    for key in ['GLOBULE_EMBEDDING__PROVIDER', 'GLOBULE_EMBEDDING__MODEL', 
                'GLOBULE_STORAGE__BACKEND', 'GLOBULE_STORAGE__PATH']:
        os.environ.pop(key, None)

def example_factory_usage():
    """Using individual factories for dependency injection."""
    print("\n=== Factory Usage ===")
    
    # Create configuration
    config = PydanticConfigManager(overrides={
        'embedding': {'provider': 'ollama', 'model': 'mxbai-embed-large'},
        'storage': {'backend': 'sqlite', 'path': ':memory:'}
    })
    
    # Create individual components using factories
    try:
        # Note: These will fail without proper mocking in real usage
        # but demonstrate the factory pattern
        
        print("Creating embedding adapter...")
        # embedding_adapter = EmbeddingAdapterFactory.create(config)
        print("Embedding factory would create OllamaEmbeddingAdapter")
        
        print("Creating storage manager...")
        # storage_manager = StorageManagerFactory.create(config)  
        print("Storage factory would create SqliteStorageAdapter")
        
    except Exception as e:
        print(f"Expected error without real services: {type(e).__name__}")

def example_configuration_sections():
    """Working with configuration sections."""
    print("\n=== Configuration Sections ===")
    
    config = PydanticConfigManager(overrides={
        'embedding': {
            'provider': 'ollama',
            'model': 'mxbai-embed-large',
            'endpoint': 'https://localhost:11434'
        },
        'storage': {
            'backend': 'postgres',
            'path': 'postgresql://user:pass@localhost:5432/db'
        }
    })
    
    # Get entire sections
    embedding_section = config.get_section('embedding')
    storage_section = config.get_section('storage')
    
    print("Embedding section:")
    for key, value in embedding_section.items():
        print(f"  {key}: {value}")
    
    print("Storage section:")
    for key, value in storage_section.items():
        print(f"  {key}: {value}")

def example_configuration_validation():
    """Demonstrating configuration validation."""
    print("\n=== Configuration Validation ===")
    
    try:
        # Valid configuration
        valid_config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'model': 'mxbai-embed-large',
                'endpoint': 'https://localhost:11434'  # Must be HTTPS
            }
        })
        print("[OK] Valid configuration accepted")
        
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
    
    try:
        # Invalid configuration - HTTP endpoint
        invalid_config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'endpoint': 'http://localhost:11434'  # Invalid: must be HTTPS
            }
        })
        
    except Exception as e:
        print(f"[OK] Expected validation error caught: {type(e).__name__}")

def example_legacy_compatibility():
    """Using legacy configuration interface."""
    print("\n=== Legacy Compatibility ===")
    
    # The old interface still works
    from globule.config.settings import get_config
    
    legacy_config = get_config()
    print(f"Legacy interface - model: {legacy_config.default_embedding_model}")
    print(f"Legacy interface - storage: {legacy_config.storage_path}")
    print(f"Legacy interface - URL: {legacy_config.ollama_base_url}")
    
    # But new code should prefer the new interface
    new_config = PydanticConfigManager()
    print(f"New interface - model: {new_config.get('embedding.model')}")
    print(f"New interface - storage: {new_config.get('storage.path')}")

def example_deployment_patterns():
    """Common deployment configuration patterns."""
    print("\n=== Deployment Patterns ===")
    
    # Development pattern
    dev_config = {
        'embedding': {'provider': 'ollama', 'model': 'llama3.2:1b'},
        'storage': {'backend': 'sqlite', 'path': './dev.db'}
    }
    print("Development config:", dev_config)
    
    # Production pattern  
    prod_config = {
        'embedding': {
            'provider': 'ollama', 
            'model': 'mxbai-embed-large',
            'endpoint': 'https://ollama-prod.company.com:11434'
        },
        'storage': {
            'backend': 'postgres',
            'path': 'postgresql://app:secret@db-prod:5432/globule'
        }
    }
    print("Production config:", prod_config)
    
    # Multi-tenant pattern
    def get_tenant_config(tenant_id: str):
        return {
            'storage': {
                'backend': 'postgres',
                'path': f'postgresql://app:secret@db:5432/tenant_{tenant_id}'
            }
        }
    
    tenant_config = get_tenant_config("customer123")
    print("Tenant config:", tenant_config)

def example_configuration_debugging():
    """Debugging configuration issues."""
    print("\n=== Configuration Debugging ===")
    
    # Check current effective configuration
    config = PydanticConfigManager()
    
    print("Current effective configuration:")
    print(f"  Embedding provider: {config.get('embedding.provider')}")
    print(f"  Embedding model: {config.get('embedding.model')}")
    print(f"  Embedding endpoint: {config.get('embedding.endpoint')}")
    print(f"  Storage backend: {config.get('storage.backend')}")
    print(f"  Storage path: {config.get('storage.path')}")
    
    # Get configuration sections for detailed inspection
    print("\nDetailed sections:")
    print("Embedding:", config.get_section('embedding'))
    print("Storage:", config.get_section('storage'))
    
    # Check for missing keys with defaults
    print(f"\nNon-existent key with default: {config.get('nonexistent.key', 'DEFAULT_VALUE')}")

if __name__ == "__main__":
    """Run all examples."""
    print("Globule Configuration Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_explicit_overrides() 
        example_environment_variables()
        example_factory_usage()
        example_configuration_sections()
        example_configuration_validation()
        example_legacy_compatibility()
        example_deployment_patterns()
        example_configuration_debugging()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()