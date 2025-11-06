"""
Storage Configuration Manager
Handles local vs cloud storage modes for Headspace
"""

import os
from enum import Enum
from typing import Optional


class StorageMode(Enum):
    """Storage mode options"""
    LOCAL = "local"
    CLOUD = "cloud"  # Supabase
    AUTO = "auto"    # Auto-detect based on environment


class StorageManager:
    """Manages storage configuration and mode"""
    
    def __init__(self):
        self.mode = self._detect_storage_mode()
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.user_id = os.environ.get("USER_ID")  # For cloud mode
        
    def _detect_storage_mode(self) -> StorageMode:
        """Detect storage mode from environment or config"""
        mode_str = os.environ.get("STORAGE_MODE", "auto").lower()
        
        if mode_str == "local":
            return StorageMode.LOCAL
        elif mode_str == "cloud":
            return StorageMode.CLOUD
        else:
            # Auto-detect: use cloud if Supabase credentials are present
            if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
                return StorageMode.CLOUD
            return StorageMode.LOCAL
    
    def is_local(self) -> bool:
        """Check if using local storage"""
        return self.mode == StorageMode.LOCAL
    
    def is_cloud(self) -> bool:
        """Check if using cloud storage"""
        return self.mode == StorageMode.CLOUD
    
    def get_mode(self) -> str:
        """Get current storage mode as string"""
        return self.mode.value
    
    def can_use_cloud(self) -> bool:
        """Check if cloud storage is available"""
        return bool(self.supabase_url and self.supabase_key)

