"""
Canvas Skeleton: Template system for saving and reusing canvas layouts.

This module provides functionality to save canvas layouts as reusable templates
that capture module positions, schema types, and relationships without storing
the actual content. Users can then apply these templates to new searches.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SkeletonType(Enum):
    """Types of canvas skeletons."""
    USER_CREATED = "user_created"
    AUTO_GENERATED = "auto_generated"
    SYSTEM_DEFAULT = "system_default"


@dataclass
class ModulePlaceholder:
    """Represents a module position in the skeleton without actual content."""
    id: str
    position_key: str  # e.g., "top-left", "center", "bottom-right"
    schema_name: str
    layout_type: str  # widget, panel, fullscreen, sidebar
    size: str  # small, medium, large, auto, full
    title_template: str  # Template for module titles (e.g., "Search: {query}")
    content_template: str  # Template for module content structure
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModulePlaceholder':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CanvasSkeleton:
    """
    A canvas layout template that can be saved and reused.
    
    Contains module positions and schema information without actual content,
    allowing users to apply consistent layouts to different searches.
    """
    name: str
    description: str
    skeleton_type: SkeletonType
    created_at: datetime
    modified_at: datetime
    author: str
    tags: List[str]
    
    # Layout structure
    grid_config: Dict[str, Any]  # Grid size, gutters, etc.
    module_placeholders: List[ModulePlaceholder]
    
    # Usage metadata
    usage_count: int
    last_used: Optional[datetime]
    
    # Schema relationships
    primary_schemas: List[str]  # Main schemas this skeleton works best with
    compatible_schemas: List[str]  # Additional compatible schemas
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skeleton to dictionary for JSON storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['modified_at'] = self.modified_at.isoformat()
        data['last_used'] = self.last_used.isoformat() if self.last_used else None
        data['skeleton_type'] = self.skeleton_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanvasSkeleton':
        """Create skeleton from dictionary."""
        # Parse datetime strings
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used']) if data['last_used'] else None
        data['skeleton_type'] = SkeletonType(data['skeleton_type'])
        
        # Parse module placeholders
        placeholders = []
        for placeholder_data in data['module_placeholders']:
            placeholders.append(ModulePlaceholder.from_dict(placeholder_data))
        data['module_placeholders'] = placeholders
        
        return cls(**data)
    
    def get_modules_by_position(self) -> Dict[str, List[ModulePlaceholder]]:
        """Group module placeholders by position."""
        by_position = {}
        for placeholder in self.module_placeholders:
            pos = placeholder.position_key
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(placeholder)
        return by_position
    
    def get_modules_by_schema(self) -> Dict[str, List[ModulePlaceholder]]:
        """Group module placeholders by schema."""
        by_schema = {}
        for placeholder in self.module_placeholders:
            schema = placeholder.schema_name
            if schema not in by_schema:
                by_schema[schema] = []
            by_schema[schema].append(placeholder)
        return by_schema
    
    def is_compatible_with_schemas(self, schema_names: List[str]) -> bool:
        """Check if skeleton is compatible with given schemas."""
        available_schemas = set(self.primary_schemas + self.compatible_schemas)
        required_schemas = set(placeholder.schema_name for placeholder in self.module_placeholders)
        return required_schemas.issubset(available_schemas) or any(s in schema_names for s in available_schemas)
    
    def update_usage(self):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()
        self.modified_at = datetime.now()


class SkeletonManager:
    """
    Manages saving, loading, and organizing canvas skeletons.
    
    Provides functionality to create skeleton templates from existing canvas layouts
    and apply them to new canvases.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize skeleton manager.
        
        Args:
            storage_dir: Directory to store skeleton files (default: ~/.globule/skeletons)
        """
        self.storage_dir = storage_dir or Path.home() / '.globule' / 'skeletons'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded skeletons
        self._skeleton_cache: Dict[str, CanvasSkeleton] = {}
        self._load_existing_skeletons()
    
    def _load_existing_skeletons(self):
        """Load all existing skeletons from storage."""
        try:
            for skeleton_file in self.storage_dir.glob('*.json'):
                try:
                    skeleton = self.load_skeleton(skeleton_file.stem)
                    if skeleton:
                        self._skeleton_cache[skeleton.name] = skeleton
                except Exception as e:
                    logger.warning(f"Failed to load skeleton {skeleton_file.stem}: {e}")
        except Exception as e:
            logger.error(f"Failed to load skeletons from {self.storage_dir}: {e}")
    
    def create_skeleton_from_canvas(self, 
                                   canvas_modules: List[Any],  # CanvasModule objects
                                   name: str,
                                   description: str,
                                   author: str = "user",
                                   tags: Optional[List[str]] = None) -> CanvasSkeleton:
        """
        Create a skeleton template from existing canvas modules.
        
        Args:
            canvas_modules: List of CanvasModule objects from current canvas
            name: Name for the skeleton
            description: Description of the skeleton's purpose
            author: Creator of the skeleton
            tags: Optional tags for categorization
            
        Returns:
            CanvasSkeleton object
        """
        now = datetime.now()
        tags = tags or []
        
        # Extract module placeholders from canvas modules
        placeholders = []
        schemas_used = set()
        
        for i, module in enumerate(canvas_modules):
            # Generate content templates based on module content structure
            title_template = f"Module {i+1}: {{query}}"
            content_template = "{{content}}"  # Basic template
            
            # Try to extract more specific templates based on content
            if hasattr(module, 'content') and module.content:
                content = str(module.content)
                if '|' in content and '---' in content:
                    content_template = "## {title}\n\n{table_content}"
                elif content.startswith('##'):
                    content_template = "## {title}\n\n{content}"
                else:
                    content_template = "{content}"
            
            placeholder = ModulePlaceholder(
                id=f"placeholder_{i}",
                position_key=module.layout.position.value,
                schema_name=module.layout.schema_name,
                layout_type=module.layout.layout_type.value,
                size=module.layout.size.value,
                title_template=title_template,
                content_template=content_template,
                metadata={
                    'original_module_id': module.id,
                    'original_name': module.name
                }
            )
            placeholders.append(placeholder)
            schemas_used.add(module.layout.schema_name)
        
        # Create grid configuration (3x3 with gutter)
        grid_config = {
            'type': 'grid',
            'columns': 3,
            'rows': 3,
            'gutter': {'x': 1, 'y': 1}
        }
        
        skeleton = CanvasSkeleton(
            name=name,
            description=description,
            skeleton_type=SkeletonType.USER_CREATED,
            created_at=now,
            modified_at=now,
            author=author,
            tags=tags,
            grid_config=grid_config,
            module_placeholders=placeholders,
            usage_count=0,
            last_used=None,
            primary_schemas=list(schemas_used),
            compatible_schemas=[]
        )
        
        return skeleton
    
    def save_skeleton(self, skeleton: CanvasSkeleton) -> bool:
        """
        Save skeleton to storage.
        
        Args:
            skeleton: CanvasSkeleton to save
            
        Returns:
            True if saved successfully
        """
        try:
            skeleton_path = self.storage_dir / f"{skeleton.name}.json"
            
            with open(skeleton_path, 'w', encoding='utf-8') as f:
                json.dump(skeleton.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._skeleton_cache[skeleton.name] = skeleton
            
            logger.info(f"Saved skeleton '{skeleton.name}' to {skeleton_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save skeleton '{skeleton.name}': {e}")
            return False
    
    def load_skeleton(self, name: str) -> Optional[CanvasSkeleton]:
        """
        Load skeleton from storage.
        
        Args:
            name: Name of the skeleton to load
            
        Returns:
            CanvasSkeleton object or None if not found
        """
        # Check cache first
        if name in self._skeleton_cache:
            return self._skeleton_cache[name]
        
        try:
            skeleton_path = self.storage_dir / f"{name}.json"
            if not skeleton_path.exists():
                return None
            
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            skeleton = CanvasSkeleton.from_dict(data)
            self._skeleton_cache[name] = skeleton
            
            return skeleton
            
        except Exception as e:
            logger.error(f"Failed to load skeleton '{name}': {e}")
            return None
    
    def list_skeletons(self) -> List[CanvasSkeleton]:
        """Get list of all available skeletons."""
        return list(self._skeleton_cache.values())
    
    def search_skeletons(self, 
                        query: Optional[str] = None,
                        schemas: Optional[List[str]] = None,
                        tags: Optional[List[str]] = None,
                        skeleton_type: Optional[SkeletonType] = None) -> List[CanvasSkeleton]:
        """
        Search skeletons by various criteria.
        
        Args:
            query: Text search in name/description
            schemas: Filter by compatible schemas
            tags: Filter by tags
            skeleton_type: Filter by skeleton type
            
        Returns:
            List of matching skeletons
        """
        results = list(self._skeleton_cache.values())
        
        if query:
            query_lower = query.lower()
            results = [s for s in results 
                      if query_lower in s.name.lower() or query_lower in s.description.lower()]
        
        if schemas:
            results = [s for s in results if s.is_compatible_with_schemas(schemas)]
        
        if tags:
            results = [s for s in results if any(tag in s.tags for tag in tags)]
        
        if skeleton_type:
            results = [s for s in results if s.skeleton_type == skeleton_type]
        
        # Sort by usage count and last used
        results.sort(key=lambda x: (x.usage_count, x.last_used or datetime.min), reverse=True)
        
        return results
    
    def delete_skeleton(self, name: str) -> bool:
        """
        Delete skeleton from storage.
        
        Args:
            name: Name of skeleton to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            skeleton_path = self.storage_dir / f"{name}.json"
            if skeleton_path.exists():
                skeleton_path.unlink()
            
            # Remove from cache
            if name in self._skeleton_cache:
                del self._skeleton_cache[name]
            
            logger.info(f"Deleted skeleton '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete skeleton '{name}': {e}")
            return False
    
    def apply_skeleton_to_queries(self, 
                                 skeleton: CanvasSkeleton,
                                 query_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply skeleton template to new query results.
        
        Args:
            skeleton: Skeleton template to apply
            query_results: Results from searches/queries to fill template
            
        Returns:
            List of module configurations ready for canvas
        """
        modules = []
        
        for placeholder in skeleton.module_placeholders:
            # Find matching query result for this placeholder
            # Safe format to avoid conflicts with query_results keys
            try:
                title = placeholder.title_template.format(query=query_results.get('query', 'Query'))
            except (KeyError, ValueError):
                title = placeholder.title_template.replace('{query}', str(query_results.get('query', 'Query')))
            
            try:
                content = placeholder.content_template.format(content=query_results.get('content', 'No content'))
            except (KeyError, ValueError):
                content = placeholder.content_template.replace('{content}', str(query_results.get('content', 'No content')))
            
            module_config = {
                'id': f"applied_{placeholder.id}",
                'position': placeholder.position_key,
                'schema_name': placeholder.schema_name,
                'layout_type': placeholder.layout_type,
                'size': placeholder.size,
                'title': title,
                'content': content,
                'metadata': placeholder.metadata.copy()
            }
            
            modules.append(module_config)
        
        # Update skeleton usage
        skeleton.update_usage()
        self.save_skeleton(skeleton)
        
        return modules
    
    def get_skeleton_stats(self) -> Dict[str, Any]:
        """Get statistics about skeleton collection."""
        skeletons = list(self._skeleton_cache.values())
        
        if not skeletons:
            return {'total': 0}
        
        stats = {
            'total': len(skeletons),
            'by_type': {},
            'by_schema': {},
            'total_usage': sum(s.usage_count for s in skeletons),
            'most_used': max(skeletons, key=lambda x: x.usage_count).name,
            'newest': max(skeletons, key=lambda x: x.created_at).name,
            'average_modules_per_skeleton': sum(len(s.module_placeholders) for s in skeletons) / len(skeletons)
        }
        
        # Count by type
        for skeleton in skeletons:
            skeleton_type = skeleton.skeleton_type.value
            stats['by_type'][skeleton_type] = stats['by_type'].get(skeleton_type, 0) + 1
        
        # Count by schema
        for skeleton in skeletons:
            for schema in skeleton.primary_schemas:
                stats['by_schema'][schema] = stats['by_schema'].get(schema, 0) + 1
        
        return stats