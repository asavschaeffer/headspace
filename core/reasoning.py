from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class Decision:
    """
    Represents a proposed change to the filesystem.
    """
    action: str  # e.g., "move", "delete", "merge", "rename"
    target_path: str
    destination_path: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReasoningStrategy(ABC):
    """
    Abstract base class for reasoning strategies.
    Strategies propose decisions based on file metadata.
    """
    
    @abstractmethod
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        """
        Analyze file metadata and propose decisions.
        
        Args:
            file_metadata: List of metadata dictionaries for files in scope.
            
        Returns:
            List of Decision objects.
        """
        pass

class DecisionMaker:
    """
    Aggregates reasoning strategies and produces a final list of decisions.
    """
    
    def __init__(self):
        self.strategies: List[ReasoningStrategy] = []
        
    def add_strategy(self, strategy: ReasoningStrategy):
        self.strategies.append(strategy)
        
    def make_decisions(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        all_decisions = []
        for strategy in self.strategies:
            decisions = strategy.reason(file_metadata)
            all_decisions.extend(decisions)
            
        # Future: Conflict resolution, de-duplication, etc.
        return all_decisions
