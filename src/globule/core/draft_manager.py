"""
Draft Manager: Shared draft logic for CLI and TUI interfaces.

This module provides a unified interface for managing draft files,
allowing both CLI and TUI to work with the same draft state.
Keeps draft operations consistent between interfaces.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DraftManager:
    """Manages draft files with consistent operations between CLI and TUI."""
    
    def __init__(self, draft_path: str = "drafts/current_draft.md"):
        """
        Initialize draft manager with a specific draft file.
        
        Args:
            draft_path: Path to the draft file (default: drafts/current_draft.md)
        """
        self.draft_path = draft_path
        
        # Ensure drafts directory exists
        draft_dir = os.path.dirname(self.draft_path)
        if draft_dir:  # Only create if there's a directory part
            os.makedirs(draft_dir, exist_ok=True)
        
        # Initialize draft file if it doesn't exist
        if not os.path.exists(self.draft_path):
            self._initialize_draft_file()
    
    def _initialize_draft_file(self):
        """Initialize a new draft file with header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        initial_content = f"""# Draft

**Created:** {timestamp}
**Tool:** Globule CLI/TUI

## Contents

"""
        with open(self.draft_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)
        logger.info(f"Initialized new draft file: {self.draft_path}")
    
    def add_to_draft(self, content: str, section_title: Optional[str] = None):
        """
        Add content to the draft file.
        
        Args:
            content: Content to add to the draft
            section_title: Optional section title to add before content
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Prepare the content block
        content_block = ""
        if section_title:
            content_block += f"## {section_title}\n\n"
        
        content_block += f"{content}\n\n"
        content_block += f"*Added at {timestamp}*\n\n---\n\n"
        
        # Append to draft file
        with open(self.draft_path, 'a', encoding='utf-8') as f:
            f.write(content_block)
        
        logger.info(f"Added content to {self.draft_path}")
        
        return len(content_block)  # Return bytes added for confirmation
    
    def get_draft_content(self) -> str:
        """
        Get the current draft content.
        
        Returns:
            The full content of the draft file
        """
        try:
            with open(self.draft_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Draft file not found: {self.draft_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading draft file: {e}")
            return ""
    
    def get_draft_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current draft.
        
        Returns:
            Dictionary with draft statistics
        """
        content = self.get_draft_content()
        if not content:
            return {"exists": False, "size": 0, "lines": 0, "words": 0}
        
        lines = content.count('\n') + 1
        words = len(content.split())
        size = len(content.encode('utf-8'))
        
        return {
            "exists": True,
            "path": self.draft_path,
            "size": size,
            "lines": lines,
            "words": words,
            "modified": datetime.fromtimestamp(os.path.getmtime(self.draft_path)).isoformat()
        }
    
    def export_draft(self, output_path: str, format: str = "md") -> bool:
        """
        Export the draft to another file in the specified format.
        
        Args:
            output_path: Path for the exported file
            format: Export format ('md' for now, extensible)
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            content = self.get_draft_content()
            if not content:
                logger.warning("No draft content to export")
                return False
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if format == "md":
                # Add export metadata
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                export_header = f"""---
exported: {timestamp}
source: {self.draft_path}
format: markdown
---

"""
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(export_header + content)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported draft to {output_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def clear_draft(self) -> bool:
        """
        Clear the current draft (reinitialize it).
        
        Returns:
            True if clear succeeded, False otherwise
        """
        try:
            self._initialize_draft_file()
            logger.info(f"Cleared draft: {self.draft_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear draft: {e}")
            return False
    
    def backup_draft(self, backup_suffix: Optional[str] = None) -> str:
        """
        Create a backup of the current draft.
        
        Args:
            backup_suffix: Optional suffix for backup filename
            
        Returns:
            Path to the backup file
        """
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create backup filename
        base_name = os.path.splitext(self.draft_path)[0]
        backup_path = f"{base_name}_backup_{backup_suffix}.md"
        
        try:
            content = self.get_draft_content()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Created draft backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise