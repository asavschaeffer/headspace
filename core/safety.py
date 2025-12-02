import os
from pathlib import Path
from typing import List, Tuple
from core.reasoning import Decision

class SafetyGuard:
    """
    Ensures file operations are safe and confined.
    """
    
    CRITICAL_DIRS = {'.git', '.venv', 'venv', '.gemini', '__pycache__', 'node_modules'}
    CRITICAL_FILES = {'.env', 'transaction_log.json', 'file_index.db', 'search_index.pkl'}
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        
    def validate_decision(self, decision: Decision) -> Tuple[bool, str]:
        """
        Checks if a decision is safe to execute.
        Returns (is_safe, reason).
        """
        try:
            target = Path(decision.target_path).resolve()
            
            # 1. Check confinement
            if not self._is_confined(target):
                return False, f"Target {target} is outside workspace {self.workspace_root}"
                
            if decision.destination_path:
                dest = Path(decision.destination_path).resolve()
                if not self._is_confined(dest):
                    return False, f"Destination {dest} is outside workspace {self.workspace_root}"
            
            # 2. Check critical files/dirs
            if self._is_critical(target):
                return False, f"Target {target.name} is a critical system file/directory."
                
            if decision.destination_path:
                dest = Path(decision.destination_path).resolve()
                if self._is_critical(dest):
                     return False, f"Destination {dest.name} is a critical system file/directory."
            
            # 3. Check for overwrite (unless explicit merge/delete)
            # This is handled by FileOperationExecutor raising FileExistsError, but we can warn here.
            # We'll leave it to the executor for now as it's not strictly "unsafe" if handled correctly.
            
            return True, "Safe"
            
        except Exception as e:
            return False, f"Validation error: {e}"
            
    def _is_confined(self, path: Path) -> bool:
        try:
            path.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False
            
    def _is_critical(self, path: Path) -> bool:
        # Check if path is or is inside a critical directory
        for part in path.parts:
            if part in self.CRITICAL_DIRS:
                return True
        
        # Check if path matches a critical file
        if path.name in self.CRITICAL_FILES:
            return True
            
        # Check if it's a directory and contains critical files/dirs
        # This prevents moving/deleting directories that contain critical system files
        if path.is_dir():
            for item in self.CRITICAL_DIRS | self.CRITICAL_FILES:
                if (path / item).exists():
                    return True
            
        return False
