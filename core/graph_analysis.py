from typing import List, Dict, Any
from pathlib import Path
from core.reasoning import ReasoningStrategy, Decision
from core.relationships import RelationshipDetector

class DependencyGraphReasoner(ReasoningStrategy):
    """
    Reasoning strategy that analyzes file dependencies to propose organization changes.
    """
    
    def __init__(self):
        self.detector = RelationshipDetector()
        
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        decisions = []
        relationships = self.detector.detect_relationships(file_metadata)
        
        for source, targets in relationships.items():
            source_path = Path(source)
            source_dir = source_path.parent
            
            for target in targets:
                target_path = Path(target)
                target_dir = target_path.parent
                
                # Heuristic: If a file imports another file from a different directory,
                # and that directory is not a common parent (simple check), propose grouping.
                # For now, we just check if they are in different directories.
                if source_dir != target_dir:
                    # We propose moving the target to the source directory?
                    # Or source to target?
                    # Or both to a new module?
                    # Let's propose moving the target to the source directory for now,
                    # assuming 'source' is the main file using 'target'.
                    # This is a weak heuristic but sufficient for the prototype.
                    
                    decisions.append(Decision(
                        action="group_files",
                        target_path=str(target_path),
                        destination_path=str(source_dir),
                        reasoning=f"File {target_path.name} is imported by {source_path.name} but is in a different directory.",
                        confidence=0.6,
                        metadata={
                            "source": str(source_path),
                            "dependency": str(target_path)
                        }
                    ))
                    
        return decisions

class CouplingAnalyzer:
    """
    Analyzes coupling between files.
    """
    
    def __init__(self):
        self.detector = RelationshipDetector()
        
    def analyze_coupling(self, file_metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        relationships = self.detector.detect_relationships(file_metadata)
        coupling_stats = {}
        
        # Initialize stats
        for meta in file_metadata:
            path = meta['path']
            coupling_stats[path] = {
                "fan_out": 0,
                "fan_in": 0,
                "outgoing": [],
                "incoming": []
            }
            
        # Calculate Fan-out (dependencies)
        for source, targets in relationships.items():
            if source in coupling_stats:
                coupling_stats[source]["fan_out"] = len(targets)
                coupling_stats[source]["outgoing"] = targets
                
                # Calculate Fan-in (dependents)
                for target in targets:
                    if target in coupling_stats:
                        coupling_stats[target]["fan_in"] += 1
                        coupling_stats[target]["incoming"].append(source)
                        
        # Calculate Instability: I = Fan-out / (Fan-in + Fan-out)
        # I = 0: Stable (no dependencies)
        # I = 1: Unstable (no dependents)
        for path, stats in coupling_stats.items():
            total = stats["fan_in"] + stats["fan_out"]
            stats["instability"] = stats["fan_out"] / total if total > 0 else 0.0
            
        return coupling_stats

class CouplingReasoner(ReasoningStrategy):
    """
    Proposes decisions based on coupling metrics.
    """
    
    def __init__(self):
        self.analyzer = CouplingAnalyzer()
        
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        decisions = []
        stats = self.analyzer.analyze_coupling(file_metadata)
        
        for path, stat in stats.items():
            # Heuristic: If a file is highly unstable (many dependencies) and has high fan-out,
            # maybe it should be moved closer to its dependencies?
            # Or if a file has high fan-in (many dependents), it's a core utility.
            
            # Let's identify "God Objects" (High Fan-In + High Fan-Out) - simplistic view
            if stat["fan_in"] > 5 and stat["fan_out"] > 5:
                decisions.append(Decision(
                    action="refactor",
                    target_path=path,
                    reasoning=f"File has high coupling (Fan-in: {stat['fan_in']}, Fan-out: {stat['fan_out']}). Consider splitting.",
                    confidence=0.7,
                    metadata=stat
                ))
                
            # Identify "Lonely" files (0 coupling) - maybe dead code?
            if stat["fan_in"] == 0 and stat["fan_out"] == 0:
                 decisions.append(Decision(
                    action="delete", # or archive
                    target_path=path,
                    reasoning="File appears to be unused and has no dependencies (Lonely).",
                    confidence=0.3, # Low confidence, might be an entry point
                    metadata=stat
                ))
                
        return decisions
