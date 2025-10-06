#!/usr/bin/env python3
"""
Centralized Configuration Manager for Headspace System
Handles all API keys, model configurations, and fallback hierarchies
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ProviderType(Enum):
    """Available provider types"""
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    MOCK = "mock"

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: ProviderType
    api_key: Optional[str] = None
    url: Optional[str] = None
    enabled: bool = True
    fallback_priority: int = 1

@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    provider_type: ProviderType
    models: List[ModelConfig]
    enabled: bool = True
    fallback_priority: int = 1

class ConfigManager:
    """
    Centralized configuration manager that handles:
    - Environment variable loading
    - API key management
    - Model configuration
    - Fallback hierarchies
    """
    
    def __init__(self, config_file: str = "loom_config.json", env_file: str = ".env"):
        self.config_file = config_file
        self.env_file = env_file
        self.config = {}
        self.env_vars = {}
        
        # Load configuration
        self._load_env_vars()
        self._load_config_file()
        self._validate_config()
        
        # Initialize provider configurations
        self._init_providers()
    
    def _load_env_vars(self):
        """Load environment variables from .env file and system"""
        env_path = Path(self.env_file)
        
        # Load from .env file if it exists
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self.env_vars[key.strip()] = value.strip()
        
        # Override with system environment variables
        for key in os.environ:
            if key.startswith(('GEMINI_', 'OPENAI_', 'OLLAMA_', 'HEADSPACE_')):
                self.env_vars[key] = os.environ[key]
    
    def _load_config_file(self):
        """Load configuration from JSON file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print(f"⚠️  Config file {self.config_file} not found, using defaults")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "preferred_provider": "ollama",
                "fallback_chain": ["ollama", "gemini", "openai", "mock"],
                "providers": {
                    "ollama": {
                        "enabled": True,
                        "url": "http://localhost:11434/api/generate",
                        "models": [
                            {"name": "gemma2:2b", "priority": 1, "enabled": True},
                            {"name": "gemma2:4b", "priority": 2, "enabled": True},
                            {"name": "qwen2:8b", "priority": 3, "enabled": False}
                        ]
                    },
                    "gemini": {
                        "enabled": True,
                        "models": [
                            {"name": "gemini-2.0-flash-exp", "priority": 1, "enabled": True},
                            {"name": "gemini-1.5-pro", "priority": 2, "enabled": False}
                        ]
                    },
                    "openai": {
                        "enabled": False,
                        "models": [
                            {"name": "gpt-4o-mini", "priority": 1, "enabled": True},
                            {"name": "gpt-4o", "priority": 2, "enabled": False}
                        ]
                    },
                    "mock": {
                        "enabled": True,
                        "models": [
                            {"name": "mock-llm", "priority": 1, "enabled": True}
                        ]
                    }
                }
            },
            "embeddings": {
                "preferred_provider": "ollama",
                "fallback_chain": ["ollama", "gemini", "sentence-transformers", "mock"],
                "providers": {
                    "ollama": {
                        "enabled": True,
                        "url": "http://localhost:11434/api/embed",
                        "models": [
                            {"name": "nomic-embed-text", "priority": 1, "enabled": True},
                            {"name": "mxbai-embed-large", "priority": 2, "enabled": False}
                        ]
                    },
                    "gemini": {
                        "enabled": True,
                        "models": [
                            {"name": "text-embedding-004", "priority": 1, "enabled": True},
                            {"name": "embedding-001", "priority": 2, "enabled": False}
                        ]
                    },
                    "sentence-transformers": {
                        "enabled": True,
                        "models": [
                            {"name": "all-MiniLM-L6-v2", "priority": 1, "enabled": True},
                            {"name": "all-mpnet-base-v2", "priority": 2, "enabled": False}
                        ]
                    },
                    "mock": {
                        "enabled": True,
                        "models": [
                            {"name": "mock-embedding", "priority": 1, "enabled": True}
                        ]
                    }
                }
            }
        }
    
    def _validate_config(self):
        """Validate configuration and apply environment overrides"""
        # Override API keys from environment
        if "GEMINI_API_KEY" in self.env_vars:
            if "api_keys" not in self.config:
                self.config["api_keys"] = {}
            self.config["api_keys"]["gemini"] = self.env_vars["GEMINI_API_KEY"]
        
        if "OPENAI_API_KEY" in self.env_vars:
            if "api_keys" not in self.config:
                self.config["api_keys"] = {}
            self.config["api_keys"]["openai"] = self.env_vars["OPENAI_API_KEY"]
        
        # Override Ollama URL if provided (for both LLM and embeddings)
        if "OLLAMA_URL" in self.env_vars:
            ollama_base_url = self.env_vars["OLLAMA_URL"]

            # Update LLM provider URL
            if "llm" not in self.config:
                self.config["llm"] = self._get_default_config()["llm"]
            if "providers" not in self.config["llm"]:
                self.config["llm"]["providers"] = {}
            if "ollama" not in self.config["llm"]["providers"]:
                self.config["llm"]["providers"]["ollama"] = {}
            self.config["llm"]["providers"]["ollama"]["url"] = f"{ollama_base_url}/api/generate"

            # Update embedding provider URL
            if "embeddings" not in self.config:
                self.config["embeddings"] = self._get_default_config()["embeddings"]
            if "providers" not in self.config["embeddings"]:
                self.config["embeddings"]["providers"] = {}
            if "ollama" not in self.config["embeddings"]["providers"]:
                self.config["embeddings"]["providers"]["ollama"] = {}
            self.config["embeddings"]["providers"]["ollama"]["url"] = f"{ollama_base_url}/api/embed"
    
    def _init_providers(self):
        """Initialize provider configurations"""
        self.llm_providers = {}
        self.embedding_providers = {}
        
        # Initialize LLM providers
        llm_config = self.config.get("llm", {})
        for provider_name, provider_config in llm_config.get("providers", {}).items():
            if provider_config.get("enabled", True):
                models = []
                for model_config in provider_config.get("models", []):
                    if model_config.get("enabled", True):
                        api_key = None
                        if provider_name == "gemini":
                            api_key = self.config.get("api_keys", {}).get("gemini")
                        elif provider_name == "openai":
                            api_key = self.config.get("api_keys", {}).get("openai")
                        
                        models.append(ModelConfig(
                            name=model_config["name"],
                            provider=ProviderType(provider_name),
                            api_key=api_key,
                            url=provider_config.get("url"),
                            enabled=True,
                            fallback_priority=model_config.get("priority", 1)
                        ))
                
                if models:
                    self.llm_providers[provider_name] = ProviderConfig(
                        provider_type=ProviderType(provider_name),
                        models=sorted(models, key=lambda x: x.fallback_priority),
                        enabled=True,
                        fallback_priority=provider_config.get("priority", 1)
                    )
        
        # Initialize embedding providers
        embedding_config = self.config.get("embeddings", {})
        for provider_name, provider_config in embedding_config.get("providers", {}).items():
            if provider_config.get("enabled", True):
                models = []
                for model_config in provider_config.get("models", []):
                    if model_config.get("enabled", True):
                        api_key = None
                        if provider_name == "gemini":
                            api_key = self.config.get("api_keys", {}).get("gemini")
                        
                        models.append(ModelConfig(
                            name=model_config["name"],
                            provider=ProviderType(provider_name),
                            api_key=api_key,
                            url=provider_config.get("url"),
                            enabled=True,
                            fallback_priority=model_config.get("priority", 1)
                        ))
                
                if models:
                    self.embedding_providers[provider_name] = ProviderConfig(
                        provider_type=ProviderType(provider_name),
                        models=sorted(models, key=lambda x: x.fallback_priority),
                        enabled=True,
                        fallback_priority=provider_config.get("priority", 1)
                    )
    
    def get_llm_config(self, preferred_provider: Optional[str] = None) -> ModelConfig:
        """
        Get the best available LLM configuration based on fallback chain
        Returns the first available model from the fallback chain
        """
        fallback_chain = self.config.get("llm", {}).get("fallback_chain", ["ollama", "gemini", "openai", "mock"])
        
        if preferred_provider and preferred_provider in fallback_chain:
            # Move preferred provider to front
            fallback_chain = [preferred_provider] + [p for p in fallback_chain if p != preferred_provider]
        
        for provider_name in fallback_chain:
            if provider_name in self.llm_providers:
                provider = self.llm_providers[provider_name]
                if provider.enabled and provider.models:
                    return provider.models[0]  # Return first (highest priority) model
        
        # Fallback to mock if nothing else works
        return ModelConfig(
            name="mock-llm",
            provider=ProviderType.MOCK,
            enabled=True,
            fallback_priority=999
        )
    
    def get_embedding_config(self, preferred_provider: Optional[str] = None) -> ModelConfig:
        """
        Get the best available embedding configuration based on fallback chain
        Returns the first available model from the fallback chain
        """
        fallback_chain = self.config.get("embeddings", {}).get("fallback_chain", ["ollama", "gemini", "sentence-transformers", "mock"])
        
        if preferred_provider and preferred_provider in fallback_chain:
            # Move preferred provider to front
            fallback_chain = [preferred_provider] + [p for p in fallback_chain if p != preferred_provider]
        
        for provider_name in fallback_chain:
            if provider_name in self.embedding_providers:
                provider = self.embedding_providers[provider_name]
                if provider.enabled and provider.models:
                    return provider.models[0]  # Return first (highest priority) model
        
        # Fallback to mock if nothing else works
        return ModelConfig(
            name="mock-embedding",
            provider=ProviderType.MOCK,
            enabled=True,
            fallback_priority=999
        )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        return self.config.get("api_keys", {}).get(provider)
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available and configured"""
        if provider == "ollama":
            # Check if Ollama is running
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=2)
                return response.status_code == 200
            except:
                return False
        elif provider == "gemini":
            return bool(self.get_api_key("gemini")) and self.get_api_key("gemini") != "YOUR_GEMINI_API_KEY"
        elif provider == "openai":
            return bool(self.get_api_key("openai")) and self.get_api_key("openai") != "YOUR_OPENAI_API_KEY"
        elif provider in ["sentence-transformers", "mock"]:
            return True
        return False
    
    def get_all_available_providers(self) -> Dict[str, List[str]]:
        """Get all available providers and their models"""
        available = {"llm": [], "embeddings": []}
        
        for provider_name, provider in self.llm_providers.items():
            if self.is_provider_available(provider_name):
                available["llm"].extend([model.name for model in provider.models])
        
        for provider_name, provider in self.embedding_providers.items():
            if self.is_provider_available(provider_name):
                available["embeddings"].extend([model.name for model in provider.models])
        
        return available
    
    def print_status(self):
        """Print configuration status"""
        print("🔧 Configuration Status:")
        print("=" * 50)
        
        print("\n📡 LLM Providers:")
        for provider_name, provider in self.llm_providers.items():
            status = "✅" if self.is_provider_available(provider_name) else "❌"
            print(f"  {status} {provider_name}: {[m.name for m in provider.models]}")
        
        print("\n🧠 Embedding Providers:")
        for provider_name, provider in self.embedding_providers.items():
            status = "✅" if self.is_provider_available(provider_name) else "❌"
            print(f"  {status} {provider_name}: {[m.name for m in provider.models]}")
        
        print("\n🔑 API Keys:")
        for provider in ["gemini", "openai"]:
            key = self.get_api_key(provider)
            if key and key not in [f"YOUR_{provider.upper()}_API_KEY", ""]:
                print(f"  ✅ {provider}: configured")
            else:
                print(f"  ❌ {provider}: not configured")
        
        print("\n🎯 Active Configurations:")
        llm_config = self.get_llm_config()
        embedding_config = self.get_embedding_config()
        print(f"  LLM: {llm_config.provider.value} - {llm_config.name}")
        print(f"  Embeddings: {embedding_config.provider.value} - {embedding_config.name}")

if __name__ == "__main__":
    # Test the configuration manager
    config = ConfigManager()
    config.print_status()
    
    print("\n🧪 Testing fallback chains:")
    print(f"LLM config: {config.get_llm_config()}")
    print(f"Embedding config: {config.get_embedding_config()}")

