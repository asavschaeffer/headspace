"""
Cross-platform configuration file path resolution.

Handles OS-appropriate paths for system and user configuration files
following XDG standards on Linux/macOS and Windows conventions.
"""
import os
import platform
from pathlib import Path
from typing import Optional


def system_config_path() -> Path:
    """
    Get the system-wide configuration file path.
    
    Returns:
        Path to system config file:
        - Linux/macOS: /etc/globule/config.yaml
        - Windows: C:\\ProgramData\\Globule\\config.yaml
    """
    system = platform.system()
    if system == "Windows":
        # Use ProgramData for system-wide config on Windows
        program_data = os.environ.get("PROGRAMDATA", "C:\\ProgramData")
        return Path(program_data) / "Globule" / "config.yaml"
    else:
        # Linux/macOS use /etc for system config
        return Path("/etc/globule/config.yaml")


def user_config_path() -> Path:
    """
    Get the user-specific configuration file path.
    
    Returns:
        Path to user config file:
        - Linux/macOS: $XDG_CONFIG_HOME/globule/config.yaml or ~/.config/globule/config.yaml
        - Windows: %APPDATA%\\Globule\\config.yaml
    """
    system = platform.system()
    if system == "Windows":
        # Use APPDATA for user config on Windows
        appdata = os.environ.get("APPDATA")
        if not appdata:
            # Fallback to user profile if APPDATA not set
            appdata = os.path.expanduser("~\\AppData\\Roaming")
        return Path(appdata) / "Globule" / "config.yaml"
    else:
        # Linux/macOS use XDG_CONFIG_HOME or ~/.config
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "globule" / "config.yaml"
        else:
            return Path.home() / ".config" / "globule" / "config.yaml"