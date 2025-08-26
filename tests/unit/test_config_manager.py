"""
Unit tests for configuration system layer (defaults + system file).

Tests path resolution, YAML loading, deep merge, and manager initialization.
"""
import os
import platform
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from globule.config.paths import system_config_path, user_config_path
from globule.config.sources import load_yaml_file, deep_merge
from globule.config.manager import PydanticConfigManager
from globule.config.errors import ConfigError, ConfigFileError, ConfigValidationError


class TestConfigPaths:
    """Test cross-platform configuration path resolution."""
    
    @patch('platform.system')
    def test_system_config_path_windows(self, mock_system):
        """Test system config path on Windows."""
        mock_system.return_value = "Windows"
        
        with patch.dict(os.environ, {'PROGRAMDATA': 'C:\\ProgramData'}):
            path = system_config_path()
            assert path == Path("C:\\ProgramData\\Globule\\config.yaml")
    
    @patch('platform.system')
    def test_system_config_path_linux(self, mock_system):
        """Test system config path on Linux."""
        mock_system.return_value = "Linux"
        path = system_config_path()
        assert path == Path("/etc/globule/config.yaml")
    
    @patch('platform.system')
    def test_system_config_path_macos(self, mock_system):
        """Test system config path on macOS."""
        mock_system.return_value = "Darwin"
        path = system_config_path()
        assert path == Path("/etc/globule/config.yaml")
    
    @patch('platform.system')
    def test_user_config_path_windows(self, mock_system):
        """Test user config path on Windows."""
        mock_system.return_value = "Windows"
        
        with patch.dict(os.environ, {'APPDATA': 'C:\\Users\\test\\AppData\\Roaming'}):
            path = user_config_path()
            assert path == Path("C:\\Users\\test\\AppData\\Roaming\\Globule\\config.yaml")
    
    @patch('platform.system')
    def test_user_config_path_xdg(self, mock_system):
        """Test user config path with XDG_CONFIG_HOME set."""
        mock_system.return_value = "Linux"
        
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/custom/config'}):
            path = user_config_path()
            assert path == Path("/custom/config/globule/config.yaml")
    
    @patch('platform.system')
    @patch('pathlib.Path.home')
    def test_user_config_path_fallback(self, mock_home, mock_system):
        """Test user config path fallback to ~/.config."""
        mock_system.return_value = "Linux"
        mock_home.return_value = Path("/home/test")
        
        # Clear XDG_CONFIG_HOME
        with patch.dict(os.environ, {}, clear=True):
            path = user_config_path()
            assert path == Path("/home/test/.config/globule/config.yaml")


class TestYamlLoading:
    """Test YAML file loading functionality."""
    
    def test_load_existing_yaml(self):
        """Test loading valid YAML file."""
        yaml_content = """
embedding:
  provider: huggingface
  model: custom-model
storage:
  backend: postgres
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = load_yaml_file(temp_path)
            assert result['embedding']['provider'] == 'huggingface'
            assert result['embedding']['model'] == 'custom-model'
            assert result['storage']['backend'] == 'postgres'
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns empty dict."""
        result = load_yaml_file('/nonexistent/file.yaml')
        assert result == {}
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises ConfigFileError."""
        invalid_yaml = """
embedding:
  provider: huggingface
  model: [unclosed list
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigFileError, match="Invalid YAML"):
                load_yaml_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_empty_yaml(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            result = load_yaml_file(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)


class TestDeepMerge:
    """Test deep dictionary merging functionality."""
    
    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = deep_merge(base, override)
        
        assert result == {'a': 1, 'b': 3, 'c': 4}
        # Ensure original dicts are not modified
        assert base == {'a': 1, 'b': 2}
        assert override == {'b': 3, 'c': 4}
    
    def test_nested_dict_merge(self):
        """Test deep merging of nested dictionaries."""
        base = {
            'embedding': {
                'provider': 'ollama',
                'model': 'default-model',
                'timeout': 30
            },
            'storage': {
                'backend': 'sqlite'
            }
        }
        override = {
            'embedding': {
                'model': 'custom-model',
                'endpoint': 'https://api.example.com'
            }
        }
        
        result = deep_merge(base, override)
        
        expected = {
            'embedding': {
                'provider': 'ollama',  # preserved from base
                'model': 'custom-model',  # overridden
                'timeout': 30,  # preserved from base
                'endpoint': 'https://api.example.com'  # added from override
            },
            'storage': {
                'backend': 'sqlite'  # preserved from base
            }
        }
        assert result == expected
    
    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {'items': [1, 2, 3], 'other': 'value'}
        override = {'items': [4, 5]}
        result = deep_merge(base, override)
        
        assert result == {'items': [4, 5], 'other': 'value'}
    
    def test_empty_dicts(self):
        """Test merging with empty dictionaries."""
        base = {'a': 1}
        result = deep_merge(base, {})
        assert result == {'a': 1}
        
        result = deep_merge({}, base)
        assert result == {'a': 1}
        
        result = deep_merge({}, {})
        assert result == {}


class TestPydanticConfigManager:
    """Test PydanticConfigManager three-tier cascade functionality."""
    
    def test_defaults_only(self):
        """Test manager with defaults only (no system or user files)."""
        with patch('globule.config.manager.load_yaml_file', return_value={}):
            manager = PydanticConfigManager()
            
            # Test defaults are loaded
            assert manager.get('embedding.provider') == 'ollama'
            assert manager.get('embedding.model') == 'mxbai-embed-large'
            assert manager.get('storage.backend') == 'sqlite'
            assert manager.get('storage.path') == ':memory:'
    
    def test_system_override(self):
        """Test system file overriding defaults."""
        system_config = {
            'embedding': {
                'provider': 'huggingface',
                'model': 'custom-model'
            }
        }
        
        with patch('globule.config.manager.load_yaml_file', return_value=system_config):
            manager = PydanticConfigManager()
            
            # Overridden values
            assert manager.get('embedding.provider') == 'huggingface'
            assert manager.get('embedding.model') == 'custom-model'
            # Default values preserved
            assert manager.get('storage.backend') == 'sqlite'
    
    def test_user_override_system(self):
        """Test user file overriding system file."""
        def mock_load_yaml(path):
            path_str = str(path)
            if '/etc/' in path_str or 'ProgramData' in path_str:
                return {'embedding': {'provider': 'huggingface', 'model': 'system-model'}}
            elif '.config' in path_str or 'AppData' in path_str:
                return {'embedding': {'provider': 'openai', 'endpoint': 'https://api.openai.com'}}
            return {}
        
        with patch('globule.config.manager.load_yaml_file', side_effect=mock_load_yaml):
            manager = PydanticConfigManager()
            
            # User overrides system
            assert manager.get('embedding.provider') == 'openai'
            # System value preserved where no user override
            assert manager.get('embedding.model') == 'system-model'
            # User addition  
            assert str(manager.get('embedding.endpoint')) == 'https://api.openai.com/'
    
    def test_three_tier_cascade(self):
        """Test complete three-tier cascade: defaults → system → user → explicit."""
        def mock_load_yaml(path):
            path_str = str(path)
            if '/etc/' in path_str or 'ProgramData' in path_str:
                return {
                    'embedding': {'provider': 'huggingface', 'model': 'system-model'},
                    'storage': {'backend': 'postgres'}
                }
            elif '.config' in path_str or 'AppData' in path_str:
                return {
                    'embedding': {'model': 'user-model'},
                    'storage': {'path': '/user/data.db'}
                }
            return {}
        
        explicit_overrides = {'embedding': {'endpoint': 'https://explicit.com'}}
        
        with patch('globule.config.manager.load_yaml_file', side_effect=mock_load_yaml):
            manager = PydanticConfigManager(overrides=explicit_overrides)
            
            # System overrides defaults
            assert manager.get('embedding.provider') == 'huggingface'
            assert manager.get('storage.backend') == 'postgres'
            
            # User overrides system
            assert manager.get('embedding.model') == 'user-model'
            assert manager.get('storage.path') == '/user/data.db'
            
            # Explicit overrides all
            assert str(manager.get('embedding.endpoint')) == 'https://explicit.com/'
    
    def test_missing_config_files(self):
        """Test behavior with various combinations of missing config files."""
        # Test: no system, yes user
        def mock_load_yaml_no_system(path):
            path_str = str(path)
            if '/etc/' in path_str or 'ProgramData' in path_str:
                return {}  # No system file
            elif '.config' in path_str or 'AppData' in path_str:
                return {'embedding': {'provider': 'openai'}}
            return {}
        
        with patch('globule.config.manager.load_yaml_file', side_effect=mock_load_yaml_no_system):
            manager = PydanticConfigManager()
            assert manager.get('embedding.provider') == 'openai'
            assert manager.get('storage.backend') == 'sqlite'  # default
        
        # Test: yes system, no user
        def mock_load_yaml_no_user(path):
            path_str = str(path)
            if '/etc/' in path_str or 'ProgramData' in path_str:
                return {'embedding': {'provider': 'huggingface'}}
            elif '.config' in path_str or 'AppData' in path_str:
                return {}  # No user file
            return {}
        
        with patch('globule.config.manager.load_yaml_file', side_effect=mock_load_yaml_no_user):
            manager = PydanticConfigManager()
            assert manager.get('embedding.provider') == 'huggingface'
            assert manager.get('storage.backend') == 'sqlite'  # default
    
    def test_explicit_overrides(self):
        """Test explicit overrides taking highest precedence."""
        def mock_load_yaml(path):
            path_str = str(path)
            if '/etc/' in path_str or 'ProgramData' in path_str:
                return {'embedding': {'provider': 'huggingface'}}
            elif '.config' in path_str or 'AppData' in path_str:
                return {'embedding': {'provider': 'openai'}}
            return {}
        
        explicit_overrides = {'embedding': {'provider': 'ollama', 'model': 'explicit-model'}}
        
        with patch('globule.config.manager.load_yaml_file', side_effect=mock_load_yaml):
            manager = PydanticConfigManager(overrides=explicit_overrides)
            
            # Explicit override wins over all
            assert manager.get('embedding.provider') == 'ollama'  # explicit value
            assert manager.get('embedding.model') == 'explicit-model'  # explicit value
    
    def test_validation_error(self):
        """Test that invalid configuration raises ConfigValidationError."""
        invalid_config = {
            'embedding': {
                'provider': 'invalid_provider'  # Not in allowed Literal values
            }
        }
        
        with patch('globule.config.manager.load_yaml_file', return_value=invalid_config):
            with pytest.raises(ConfigValidationError):
                PydanticConfigManager()
    
    def test_get_section(self):
        """Test get_section method."""
        with patch('globule.config.manager.load_yaml_file', return_value={}):
            manager = PydanticConfigManager()
            
            embedding_section = manager.get_section('embedding')
            assert embedding_section['provider'] == 'ollama'
            assert embedding_section['model'] == 'mxbai-embed-large'
            
            # Non-existent section
            missing_section = manager.get_section('nonexistent')
            assert missing_section == {}
    
    def test_get_with_default(self):
        """Test get method with default values."""
        with patch('globule.config.manager.load_yaml_file', return_value={}):
            manager = PydanticConfigManager()
            
            # Existing key
            assert manager.get('embedding.provider') == 'ollama'
            
            # Non-existent key with default
            assert manager.get('nonexistent.key', 'default') == 'default'
            
            # Non-existent key without default
            assert manager.get('nonexistent.key') is None
    
    def test_reload(self):
        """Test reload method (basic functionality)."""
        with patch('globule.config.manager.load_yaml_file', return_value={}):
            manager = PydanticConfigManager()
            
            # Should not raise an exception
            manager.reload()