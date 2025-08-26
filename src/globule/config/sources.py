"""
Configuration source utilities including deep merge and YAML loading.

Provides safe YAML loading and deep dictionary merging for configuration cascade.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from .errors import ConfigFileError


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load YAML configuration file.
    
    Args:
        file_path: Path to YAML file to load.
        
    Returns:
        Dictionary containing loaded configuration.
        
    Raises:
        ConfigFileError: If file cannot be read or parsed.
    """
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise ConfigFileError(f"Invalid YAML in {path}: {e}", source=str(path))
    except (OSError, IOError) as e:
        raise ConfigFileError(f"Cannot read config file {path}: {e}", source=str(path))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Merge strategy:
    - Dictionaries are merged recursively
    - Lists are replaced entirely (no merging)
    - All other values are replaced
    
    Args:
        base: Base dictionary (lower precedence).
        override: Override dictionary (higher precedence).
        
    Returns:
        New dictionary with merged values.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Replace everything else (including lists)
            result[key] = value
    
    return result