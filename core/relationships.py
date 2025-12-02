from typing import List, Dict, Any, Set
from pathlib import Path

class RelationshipDetector:
    """
    Detects relationships between files based on metadata.
    """
    
    def detect_relationships(self, file_metadata_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Analyze metadata to find dependencies.
        
        Args:
            file_metadata_list: List of metadata for all files.
            
        Returns:
            Dict mapping file path to list of dependency paths.
        """
        # Build a map of module names to file paths
        module_map = {}
        for meta in file_metadata_list:
            path = Path(meta['path'])
            # Simple heuristic: filename without extension is module name
            # This is naive and assumes flat structure or unique names for now
            module_name = path.stem
            module_map[module_name] = meta['path']
            
        relationships = {}
        
        for meta in file_metadata_list:
            source_path = meta['path']
            dependencies = []
            
            # Check imports if available
            if 'imports' in meta:
                for imp in meta['imports']:
                    # Handle "from x import y" -> x is the module
                    # Handle "import x" -> x is the module
                    base_module = imp.split('.')[0]
                    if base_module in module_map:
                        target_path = module_map[base_module]
                        if target_path != source_path:
                            dependencies.append(target_path)
                            
            if dependencies:
                relationships[source_path] = dependencies
                
        return relationships
