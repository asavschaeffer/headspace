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
