"""
Tests for PydanticConfigManager using BaseSettings implementation.

Tests the proper pydantic-settings integration with environment variables
and YAML configuration sources.
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from globule.config.manager import PydanticConfigManager
from globule.config.errors import ConfigValidationError


class TestPydanticConfigManagerBaseSettings:
    """Test BaseSettings implementation of PydanticConfigManager."""
    
    def test_defaults_only(self):
        """Test manager loads defaults correctly."""
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')):
            manager = PydanticConfigManager()
            
            assert manager.get('embedding.provider') == 'ollama'
            assert manager.get('embedding.model') == 'mxbai-embed-large'
            assert manager.get('storage.backend') == 'sqlite'
            assert manager.get('storage.path') == ':memory:'
    
    def test_env_overrides(self):
        """Test environment variable overrides."""
        env_vars = {
            'GLOBULE_EMBEDDING__PROVIDER': 'openai',
            'GLOBULE_STORAGE__BACKEND': 'postgres'
        }
        
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')), \
             patch.dict(os.environ, env_vars, clear=True):
            manager = PydanticConfigManager()
            
            # Environment overrides should work
            assert manager.get('embedding.provider') == 'openai'
            assert manager.get('storage.backend') == 'postgres'
            
            # Defaults preserved where no env override
            assert manager.get('embedding.model') == 'mxbai-embed-large'
            assert manager.get('storage.path') == ':memory:'
    
    def test_explicit_overrides(self):
        """Test explicit constructor overrides."""
        overrides = {
            'embedding': {'provider': 'huggingface', 'model': 'custom-model'},
            'storage': {'path': '/custom/path'}
        }
        
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')):
            manager = PydanticConfigManager(overrides=overrides)
            
            # Explicit overrides should work
            assert manager.get('embedding.provider') == 'huggingface'
            assert manager.get('embedding.model') == 'custom-model'
            assert manager.get('storage.path') == '/custom/path'
            
            # Defaults preserved where no override
            assert manager.get('storage.backend') == 'sqlite'
    
    def test_env_and_explicit_precedence(self):
        """Test that explicit overrides beat environment variables."""
        env_vars = {
            'GLOBULE_EMBEDDING__PROVIDER': 'openai',
            'GLOBULE_STORAGE__BACKEND': 'postgres'
        }
        
        overrides = {
            'embedding': {'provider': 'huggingface'}  # Should beat env
        }
        
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')), \
             patch.dict(os.environ, env_vars, clear=True):
            manager = PydanticConfigManager(overrides=overrides)
            
            # Explicit should beat env
            assert manager.get('embedding.provider') == 'huggingface'
            # Env should work where no explicit override
            assert manager.get('storage.backend') == 'postgres'
    
    def test_get_section(self):
        """Test get_section method."""
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')):
            manager = PydanticConfigManager()
            
            embedding_section = manager.get_section('embedding')
            assert embedding_section['provider'] == 'ollama'
            assert embedding_section['model'] == 'mxbai-embed-large'
            
            # Non-existent section
            missing_section = manager.get_section('nonexistent')
            assert missing_section == {}
    
    def test_yaml_file_integration(self):
        """Test YAML file loading with real files."""
        # Create temporary YAML files
        system_yaml = """
embedding:
  provider: huggingface
  model: system-model
storage:
  backend: postgres
"""
        
        user_yaml = """
embedding:
  model: user-model
storage:
  path: /user/data.db
"""
        
        # Create temp files and get their names
        sys_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        user_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        
        try:
            # Write content and close files properly
            sys_file.write(system_yaml)
            sys_file.close()
            
            user_file.write(user_yaml)
            user_file.close()
            
            # Mock in the manager module where the paths are imported
            with patch('globule.config.manager.system_config_path', return_value=Path(sys_file.name)), \
                 patch('globule.config.manager.user_config_path', return_value=Path(user_file.name)):
                manager = PydanticConfigManager()
                
                # System file should be loaded
                assert manager.get('embedding.provider') == 'huggingface'
                assert manager.get('storage.backend') == 'postgres'
                
                # User file should override system
                assert manager.get('embedding.model') == 'user-model'
                assert manager.get('storage.path') == '/user/data.db'
                
        finally:
            # Clean up - files should be closed by now
            try:
                os.unlink(sys_file.name)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows
            try:
                os.unlink(user_file.name)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows
    
    def test_validation_error(self):
        """Test that invalid configuration raises proper error."""
        # Invalid provider value
        overrides = {
            'embedding': {'provider': 'invalid_provider'}
        }
        
        with patch('globule.config.manager.system_config_path', return_value=Path('/nonexistent/system.yaml')), \
             patch('globule.config.manager.user_config_path', return_value=Path('/nonexistent/user.yaml')):
            with pytest.raises(ConfigValidationError):
                PydanticConfigManager(overrides=overrides)