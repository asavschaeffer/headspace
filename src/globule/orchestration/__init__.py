"""
Orchestration layer for Globule.

This module contains the core orchestration engine that coordinates
business logic across the application while remaining UI-agnostic.
"""

from .engine import GlobuleOrchestrator

__all__ = ['GlobuleOrchestrator']