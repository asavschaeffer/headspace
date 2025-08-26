
"""
Enhanced Schema management with YAML support, triggers, and actions.

This module provides utilities for loading, caching, and validating
schemas (JSON/YAML) with workflow capabilities including triggers for
automatic schema detection and actions for content enrichment.
"""

import json
import logging
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache

from globule.core.interfaces import ISchemaManager

import jsonschema
from jsonschema import Draft7Validator
try:
    from pydantic import BaseModel, create_model, ValidationError as PydanticError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

logger = logging.getLogger(__name__)


class SchemaManager(ISchemaManager):
    """Enhanced schema manager with YAML support, triggers, and workflow actions."""
    
    def __init__(self):
        """Initialize schema manager with built-in and user schemas."""
        self.built_in_dir = Path(__file__).parent
        self.user_dir = Path.home() / '.globule' / 'schemas'
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validator_cache: Dict[str, Draft7Validator] = {}
        self._pydantic_cache: Dict[str, type] = {}
        
        # Ensure user directory exists
        self.user_dir.mkdir(parents=True, exist_ok=True)
        
        # Preload schemas from both directories
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load schemas from both built-in and user directories, including output schemas."""
        # Load from built-in directory (JSON only for now)
        for schema_file in self.built_in_dir.glob("*.json"):
            self._load_schema_file(schema_file)
        
        # Load regular schemas first (excluding output_ prefixed)
        for pattern in ["*.json", "*.yaml", "*.yml"]:
            for schema_file in self.user_dir.glob(pattern):
                if not schema_file.stem.startswith('output_'):
                    self._load_schema_file(schema_file)
                
        # Load output schemas with special handling
        for pattern in ["output_*.yaml", "output_*.yml"]:
            for schema_file in self.user_dir.glob(pattern):
                self._load_output_schema_file(schema_file)
    
    def _load_output_schema_file(self, schema_file: Path) -> None:
        """Load an output schema file with special handling for queries/viz/templates/exports."""
        schema_name = schema_file.stem
        try:
            # Load YAML data
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = yaml.safe_load(f)
            
            # Validate output schema structure
            self._validate_output_schema_structure(schema_data)
            
            # Store in cache with metadata indicating it's an output schema
            schema_data['_schema_type'] = 'output'
            schema_data['_file_path'] = str(schema_file)
            self._schema_cache[schema_name] = schema_data
            
            logger.info(f"Loaded output schema: {schema_name}")
            
        except Exception as e:
            logger.error(f"Failed to load output schema {schema_file}: {e}")
    
    def _validate_output_schema_structure(self, schema_data: Dict[str, Any]) -> None:
        """Validate output schema structure has required sections."""
        required_sections = ['queries', 'template']
        for section in required_sections:
            if section not in schema_data:
                raise ValueError(f"Output schema missing required section: {section}")
        
        # Validate queries structure
        if not isinstance(schema_data['queries'], list):
            raise ValueError("Output schema 'queries' must be a list")
        
        # Validate canvas_config if present (optional)
        if 'canvas_config' in schema_data:
            self._validate_canvas_config(schema_data['canvas_config'])
        
        for query in schema_data['queries']:
            if not isinstance(query, dict) or 'name' not in query:
                raise ValueError("Each query must be a dict with 'name' field")
            if 'sql' not in query and 'llm_prompt' not in query:
                raise ValueError("Each query must have either 'sql' or 'llm_prompt' field")
        
        # Validate template is string
        if not isinstance(schema_data['template'], str):
            raise ValueError("Output schema 'template' must be a string")

    def _load_schema_file(self, schema_file: Path) -> None:
        """Load a single schema file with inheritance support."""
        schema_name = schema_file.stem
        try:
            # Load based on file extension
            if schema_file.suffix.lower() in ['.yaml', '.yml']:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = yaml.safe_load(f)
            else:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
            
            # Handle inheritance
            schema_data = self._resolve_inheritance(schema_data)
            
            # Validate schema structure
            self._validate_schema_structure(schema_data)
            
            # Cache schema and validator
            self._schema_cache[schema_name] = schema_data
            self._validator_cache[schema_name] = Draft7Validator(schema_data)
            
            logger.debug(f"Loaded schema: {schema_name} from {schema_file}")
            
        except Exception as e:
            logger.error(f"Failed to load schema {schema_name} from {schema_file}: {e}")
    
    def _resolve_inheritance(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve schema inheritance by merging with base schemas."""
        if 'inherits' not in schema_data:
            return schema_data
        
        base_name = schema_data['inherits']
        base_schema = self.get_schema(base_name)
        
        if not base_schema:
            logger.warning(f"Base schema '{base_name}' not found for inheritance")
            del schema_data['inherits']
            return schema_data
        
        # Deep merge base with current (current overrides base)
        merged = self._deep_merge(base_schema.copy(), schema_data)
        if 'inherits' in merged:
            del merged['inherits']
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_schema_structure(self, schema_data: Dict[str, Any]) -> None:
        """Validate that schema has required structure for Globule."""
        required_fields = ['title', 'type', 'properties']
        for field in required_fields:
            if field not in schema_data:
                raise ValueError(f"Schema missing required field: {field}")
        
        # Validate canvas_config if present (optional)
        if 'canvas_config' in schema_data:
            self._validate_canvas_config(schema_data['canvas_config'])
    
    def _validate_canvas_config(self, canvas_config: Dict[str, Any]) -> None:
        """Validate canvas configuration structure."""
        # Layout configuration (required)
        if 'layout' not in canvas_config:
            raise ValueError("Canvas config missing required 'layout' section")
        
        layout = canvas_config['layout']
        required_layout_fields = ['type', 'position', 'size']
        for field in required_layout_fields:
            if field not in layout:
                raise ValueError(f"Canvas layout missing required field: {field}")
        
        # Validate layout type
        valid_layout_types = ['widget', 'panel', 'fullscreen', 'sidebar']
        if layout['type'] not in valid_layout_types:
            raise ValueError(f"Invalid layout type: {layout['type']}. Must be one of {valid_layout_types}")
        
        # Validate position
        valid_positions = [
            'top-left', 'top-center', 'top-right',
            'center-left', 'center', 'center-right', 
            'bottom-left', 'bottom-center', 'bottom-right',
            'sidebar-left', 'sidebar-right', 'full-width'
        ]
        if layout['position'] not in valid_positions:
            raise ValueError(f"Invalid layout position: {layout['position']}. Must be one of {valid_positions}")
        
        # Validate size
        valid_sizes = ['small', 'medium', 'large', 'auto', 'full']
        if layout['size'] not in valid_sizes:
            raise ValueError(f"Invalid layout size: {layout['size']}. Must be one of {valid_sizes}")
        
        # Validate optional TUI style config
        if 'tui_style' in canvas_config:
            tui_style = canvas_config['tui_style']
            if not isinstance(tui_style, dict):
                raise ValueError("TUI style must be a dictionary")
        
        # Validate optional web style config  
        if 'web_style' in canvas_config:
            web_style = canvas_config['web_style']
            if not isinstance(web_style, dict):
                raise ValueError("Web style must be a dictionary")
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a schema by name.
        
        Args:
            schema_name: Name of the schema (e.g., 'default', 'academic', 'technical')
            
        Returns:
            Schema dictionary or None if not found
        """
        return self._schema_cache.get(schema_name)
    
    def get_validator(self, schema_name: str) -> Optional[Draft7Validator]:
        """
        Get a validator for a schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Validator instance or None if schema not found
        """
        return self._validator_cache.get(schema_name)
    
    def validate_data(self, data: Dict[str, Any], schema_name: str) -> tuple[bool, List[str]]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        validator = self.get_validator(schema_name)
        if not validator:
            return False, [f"Schema '{schema_name}' not found"]
        
        errors = []
        try:
            # Collect all validation errors
            for error in validator.iter_errors(data):
                # Format error message with path context
                if error.absolute_path:
                    path = " -> ".join(str(p) for p in error.absolute_path)
                    errors.append(f"{path}: {error.message}")
                else:
                    errors.append(error.message)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available schema names."""
        return list(self._schema_cache.keys())
    
    def get_canvas_config(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get canvas configuration for a schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Canvas config dictionary or None if not found or no canvas config
        """
        schema = self.get_schema(schema_name)
        if schema and 'canvas_config' in schema:
            return schema['canvas_config']
        return None
    
    def get_schemas_with_canvas_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all schemas that have canvas configuration.
        
        Returns:
            Dictionary mapping schema names to their canvas configs
        """
        result = {}
        for schema_name, schema_data in self._schema_cache.items():
            if 'canvas_config' in schema_data:
                result[schema_name] = schema_data['canvas_config']
        return result
    
    def detect_schema_for_text(self, text: str) -> Optional[str]:
        """Detect the best schema for given text using triggers."""
        for schema_name, schema_data in self._schema_cache.items():
            # Skip output schemas in text detection
            if schema_data.get('_schema_type') == 'output':
                continue
            if self._matches_triggers(text, schema_data.get('triggers', [])):
                return schema_name
        return None
    
    def detect_output_schema_for_topic(self, topic: str) -> Optional[str]:
        """Detect output schema based on topic/draft intent."""
        if not topic:
            return None
            
        topic_lower = topic.lower()
        
        # Direct mapping patterns
        topic_mappings = {
            'valet': ['valet', 'parking', 'car'],
            'waitress': ['waitress', 'server', 'restaurant', 'dining'],
            'tweet': ['tweet', 'twitter', 'x', 'social'],
            'dashboard': ['dashboard', 'analytics', 'report'],
            'summary': ['summary', 'recap', 'overview']
        }
        
        # Check for output schema matches
        for schema_name, schema in self._schema_cache.items():
            if not schema.get('_schema_type') == 'output':
                continue
                
            # Try direct name match (output_valet -> valet)
            base_name = schema_name.replace('output_', '')
            if base_name in topic_lower:
                return schema_name
            
            # Try keyword mapping
            for keyword_group, keywords in topic_mappings.items():
                if keyword_group in base_name and any(kw in topic_lower for kw in keywords):
                    return schema_name
        
        # Check if topic contains words that suggest an output schema intent
        output_indicators = ['dashboard', 'report', 'summary', 'tweet', 'post', 'export']
        if any(indicator in topic_lower for indicator in output_indicators):
            # Try to match with base schema name
            for schema_name, schema in self._schema_cache.items():
                if schema.get('_schema_type') == 'output':
                    base_name = schema_name.replace('output_', '')
                    if any(word in topic_lower for word in base_name.split('_')):
                        return schema_name
        
        return None
    
    def get_output_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get output schema by name."""
        schema = self._schema_cache.get(schema_name)
        if schema and schema.get('_schema_type') == 'output':
            return schema
        return None
    
    def list_output_schemas(self) -> List[str]:
        """List all available output schema names."""
        return [name for name, schema in self._schema_cache.items() 
                if schema.get('_schema_type') == 'output']
    
    def _matches_triggers(self, text: str, triggers: List[Dict[str, Any]]) -> bool:
        """Check if text matches any of the schema triggers."""
        if not triggers:
            return False
        
        for trigger in triggers:
            trigger_type = trigger.get('type', 'keyword')
            
            if trigger_type == 'regex':
                pattern = trigger.get('pattern', '')
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            
            elif trigger_type == 'keyword':
                keywords = trigger.get('keywords', [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        return True
            
            elif trigger_type == 'contains_all':
                words = trigger.get('words', [])
                if all(word.lower() in text.lower() for word in words):
                    return True
        
        return False
    
    def apply_schema_actions(self, text: str, schema_name: str) -> Dict[str, Any]:
        """Apply schema actions to enrich text data."""
        schema = self.get_schema(schema_name)
        if not schema:
            return {'text': text, 'enriched': {}}
        
        enriched = {'text': text}
        actions = schema.get('actions', [])
        
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'extract_regex':
                field = action.get('field', 'extracted')
                pattern = action.get('pattern', '')
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    enriched[field] = match.group(1) if match.groups() else match.group(0)
            
            elif action_type == 'extract_keywords':
                field = action.get('field', 'keywords')
                keywords = action.get('keywords', [])
                found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                enriched[field] = found_keywords
            
            elif action_type == 'set_default':
                field = action.get('field')
                value = action.get('value')
                if field and field not in enriched:
                    enriched[field] = value
        
        return {
            'text': text,
            'enriched': enriched,
            'schema_applied': schema_name
        }
    
    def get_schema_for_domain(self, domain: str) -> str:
        """
        Get the best schema name for a given domain.
        
        Args:
            domain: Domain name (e.g., 'technical', 'creative', 'academic')
            
        Returns:
            Schema name to use for this domain
        """
        # Enhanced domain mappings
        domain_mappings = {
            'technical': 'technical',
            'creative': 'creative', 
            'academic': 'academic',
            'valet': 'valet',
            'parking_service': 'valet',  # Alternative domain for valet
            'computer_science': 'academic',
            'research': 'academic',
            'engineering': 'technical',
            'software': 'technical',
            'art': 'creative',
            'writing': 'creative',
            'fiction': 'creative'
        }
        
        return domain_mappings.get(domain.lower(), 'default')
    
    def format_schema_for_llm(self, schema_name: str) -> str:
        """
        Format a schema for inclusion in LLM prompts.
        
        Args:
            schema_name: Name of schema to format
            
        Returns:
            Human-readable schema description for LLM prompts
        """
        schema = self.get_schema(schema_name)
        if not schema:
            return "No schema available"
        
        # Extract key information for LLM prompts
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        formatted_parts = []
        formatted_parts.append(f"Schema: {schema.get('title', schema_name)}")
        
        if 'description' in schema:
            formatted_parts.append(f"Description: {schema['description']}")
        
        formatted_parts.append("Required fields:")
        
        for field in required:
            if field in properties:
                prop = properties[field]
                field_desc = f"  - {field}: {prop.get('description', 'No description')}"
                
                # Add type information
                if 'type' in prop:
                    field_desc += f" (type: {prop['type']})"
                
                # Add enum values if present
                if 'enum' in prop:
                    field_desc += f" [options: {', '.join(prop['enum'])}]"
                elif prop.get('type') == 'array' and 'items' in prop and 'enum' in prop['items']:
                    field_desc += f" [item options: {', '.join(prop['items']['enum'])}]"
                
                formatted_parts.append(field_desc)
        
        # Add optional fields if any
        optional_fields = [f for f in properties if f not in required]
        if optional_fields:
            formatted_parts.append("Optional fields:")
            for field in optional_fields[:5]:  # Limit to first 5 optional fields
                if field in properties:
                    prop = properties[field]
                    field_desc = f"  - {field}: {prop.get('description', 'No description')}"
                    formatted_parts.append(field_desc)
        
        return "\n".join(formatted_parts)
    
    def create_pydantic_model(self, schema_name: str) -> Optional[type]:
        """Create a Pydantic model from a schema for validation and defaults."""
        if schema_name in self._pydantic_cache:
            return self._pydantic_cache[schema_name]
        
        schema = self.get_schema(schema_name)
        if not schema:
            return None
        
        try:
            fields = {}
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for field_name, field_def in properties.items():
                field_type = self._map_json_type_to_python(field_def)
                default_value = field_def.get('default')
                
                if field_name in required and default_value is None:
                    fields[field_name] = (field_type, ...)
                else:
                    fields[field_name] = (field_type, default_value)
            
            model_name = f"{schema_name.capitalize()}Model"
            model = create_model(model_name, **fields)
            
            self._pydantic_cache[schema_name] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to create Pydantic model for {schema_name}: {e}")
            return None
    
    def _map_json_type_to_python(self, field_def: Dict[str, Any]) -> type:
        """Map JSON Schema types to Python types."""
        json_type = field_def.get('type', 'string')
        
        if isinstance(json_type, list):
            # Handle nullable types like ["string", "null"]
            if 'null' in json_type:
                non_null_types = [t for t in json_type if t != 'null']
                if len(non_null_types) == 1:
                    base_type = self._get_python_type(non_null_types[0])
                    return Optional[base_type]
                else:
                    return Optional[Union[tuple(self._get_python_type(t) for t in non_null_types)]]
            else:
                return Union[tuple(self._get_python_type(t) for t in json_type)]
        else:
            return self._get_python_type(json_type)
    
    def _get_python_type(self, json_type: str) -> type:
        """Get Python type for JSON Schema type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': List[Any],
            'object': Dict[str, Any]
        }
        return type_mapping.get(json_type, Any)
    
    def reload_schemas(self) -> None:
        """Reload all schemas from disk."""
        self._schema_cache.clear()
        self._validator_cache.clear()
        self._pydantic_cache.clear()
        self._load_schemas()
        
    def load_custom_schema(self, schema_path: Path, schema_name: Optional[str] = None) -> bool:
        """
        Load a custom schema from a file path.
        
        Args:
            schema_path: Path to JSON/YAML schema file
            schema_name: Name to register the schema under (defaults to filename)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not schema_name:
            schema_name = schema_path.stem
        
        try:
            self._load_schema_file(schema_path)
            logger.info(f"Loaded custom schema: {schema_name} from {schema_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom schema {schema_name} from {schema_path}: {e}")
            return False


# Global schema manager instance
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager


# Convenience functions for common operations
def get_schema(schema_name: str) -> Optional[Dict[str, Any]]:
    """Get a schema by name."""
    return get_schema_manager().get_schema(schema_name)


def validate_data(data: Dict[str, Any], schema_name: str) -> tuple[bool, List[str]]:
    """Validate data against a schema."""
    return get_schema_manager().validate_data(data, schema_name)


def format_schema_for_llm(schema_name: str) -> str:
    """Format a schema for LLM prompts."""
    return get_schema_manager().format_schema_for_llm(schema_name)


def get_schema_for_domain(domain: str) -> str:
    """Get the best schema for a domain."""
    return get_schema_manager().get_schema_for_domain(domain)


def detect_schema_for_text(text: str) -> Optional[str]:
    """Detect schema for text using triggers."""
    return get_schema_manager().detect_schema_for_text(text)


def apply_schema_actions(text: str, schema_name: str) -> Dict[str, Any]:
    """Apply schema actions to text."""
    return get_schema_manager().apply_schema_actions(text, schema_name)


def create_pydantic_model(schema_name: str) -> Optional[type]:
    """Create Pydantic model for schema."""
    return get_schema_manager().create_pydantic_model(schema_name)


def detect_output_schema_for_topic(topic: str) -> Optional[str]:
    """Detect output schema for topic using triggers."""
    return get_schema_manager().detect_output_schema_for_topic(topic)


def get_output_schema(schema_name: str) -> Optional[Dict[str, Any]]:
    """Get output schema by name."""
    return get_schema_manager().get_output_schema(schema_name)


def list_output_schemas() -> List[str]:
    """List all available output schema names."""
    return get_schema_manager().list_output_schemas()
