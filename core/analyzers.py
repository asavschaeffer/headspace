from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class FileAnalyzer(ABC):
    """
    Abstract base class for file analyzers.
    Analyzers are responsible for extracting metadata from files.
    """

    @abstractmethod
    def analyze(self, path: Path) -> Dict[str, Any]:
        """
        Analyze a file and return a dictionary of metadata.
        
        Args:
            path: Path to the file to analyze.
            
        Returns:
            Dict containing metadata (e.g., summary, type, topics, dependencies).
        """
        pass

class BaseFileAnalyzer(FileAnalyzer):
    """
    Basic implementation of FileAnalyzer.
    Extracts basic file stats and uses an LLM client for summarization.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def analyze(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"error": "File not found"}
            
        stats = path.stat()
        metadata = {
            "path": str(path),
            "name": path.name,
            "size": stats.st_size,
            "extension": path.suffix,
            "modified_time": stats.st_mtime,
        }
        
        # If we have an LLM client, we could add summarization here
        # For now, we keep it simple as per "start simple"
        if self.llm_client:
            # Placeholder for LLM integration
            pass
            
        return metadata

import ast

class CodeFileAnalyzer(BaseFileAnalyzer):
    """
    Analyzer for code files.
    Extracts imports, functions, and classes using AST.
    """
    
    def analyze(self, path: Path) -> Dict[str, Any]:
        metadata = super().analyze(path)
        if "error" in metadata:
            return metadata
            
        if path.suffix == '.py':
            try:
                content = path.read_text(errors='ignore')
                tree = ast.parse(content)
                
                imports = []
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                        
                metadata['imports'] = imports
                metadata['functions'] = functions
                metadata['classes'] = classes
                metadata['type'] = 'code'
                
            except Exception as e:
                metadata['ast_error'] = str(e)
                
        return metadata
