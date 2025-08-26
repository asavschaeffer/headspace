"""
Glass Engine Core Architecture - Refactored

This module has been refactored to align with the new GlobuleAPI architecture.
It remains a unified testing/tutorial/showcase system but now acts as a client
of the GlobuleAPI, creating its own sandboxed instance to operate within.
"""

import abc
import asyncio
import logging
import time
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from rich.console import Console

# Refactored Imports
from globule.core.api import GlobuleAPI
from globule.config.settings import get_config
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.services.embedding.mock_adapter import MockEmbeddingAdapter as MockEmbeddingProvider
from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter
from globule.services.parsing.ollama_parser import OllamaParser
from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.orchestration.engine import GlobuleOrchestrator


class GlassEngineMode(Enum):
    INTERACTIVE = "interactive"
    DEMO = "demo"
    DEBUG = "debug"

@dataclass
class GlassEngineMetrics:
    mode: GlassEngineMode
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    validation_status: str = "PENDING"
    error_log: List[str] = field(default_factory=list)

    def mark_complete(self) -> None:
        self.end_time = datetime.now()
        delta = self.end_time - self.start_time
        self.total_duration_ms = delta.total_seconds() * 1000
        if not self.error_log:
            self.validation_status = "PASS"
        else:
            self.validation_status = "FAIL"

    def add_error(self, error: Exception, context: str = "") -> None:
        error_msg = f"{context}: {type(error).__name__}: {str(error)}"
        self.error_log.append(error_msg)

class AbstractGlassEngine(abc.ABC):
    """Abstract base class for all Glass Engine implementations."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.logger = logging.getLogger(f"glass_engine.{self.get_mode().value}")
        self.metrics = GlassEngineMetrics(mode=self.get_mode())
        self.api: Optional[GlobuleAPI] = None
        self._temp_dir = None
        self._storage = None
        self._embedding_provider = None
        self._parsing_provider = None

    @abc.abstractmethod
    def get_mode(self) -> GlassEngineMode:
        pass

    @abc.abstractmethod
    async def execute_tutorial_flow(self) -> None:
        pass

    async def run(self) -> GlassEngineMetrics:
        self.logger.info(f"Starting Glass Engine in {self.get_mode().value} mode")
        try:
            await self._initialize_components()
            await self.execute_tutorial_flow()
            self.metrics.mark_complete()
            self.logger.info(f"Glass Engine completed successfully in {self.metrics.total_duration_ms:.1f}ms")
        except Exception as e:
            self.metrics.add_error(e, "Glass Engine execution")
            self.metrics.mark_complete()
            self.logger.error(f"Glass Engine failed: {e}")
        finally:
            await self._cleanup_components()
        return self.metrics

    async def _initialize_components(self) -> None:
        """Initializes a sandboxed Globule environment and API."""
        self.logger.debug("Initializing sandboxed Globule components")
        self._temp_dir = tempfile.mkdtemp(prefix="globule_glass_engine_")

        try:
            config = get_config()
            # Override config to use the temporary directory
            config.data_dir = self._temp_dir

            self._storage = SQLiteStorageManager(config)
            await self._storage.initialize()

            self._embedding_provider = OllamaEmbeddingProvider()
            if not await self._embedding_provider.health_check():
                self.console.print("[yellow]Warning:[/] Ollama not accessible. Using mock embeddings.")
                await self._embedding_provider.close()
                self._embedding_provider = MockEmbeddingProvider()
            embedding_adapter = OllamaEmbeddingAdapter(self._embedding_provider)

            self._parsing_provider = OllamaParser()
            parsing_adapter = OllamaParsingAdapter(self._parsing_provider)

            orchestrator = GlobuleOrchestrator(
                embedding_provider=embedding_adapter,
                parser_provider=parsing_adapter,
                storage_manager=self._storage
            )

            self.api = GlobuleAPI(storage=self._storage, orchestrator=orchestrator)
            self.logger.debug("Sandboxed component initialization completed.")

        except Exception as e:
            await self._cleanup_components()
            raise Exception(f"Component initialization failed: {e}") from e

    async def _cleanup_components(self) -> None:
        self.logger.debug("Cleaning up components")
        if self._embedding_provider: await self._embedding_provider.close()
        if self._parsing_provider: await self._parsing_provider.close()
        if self._storage: await self._storage.close()
        if self._temp_dir and shutil:
            shutil.rmtree(self._temp_dir)
            self.logger.debug(f"Removed temporary directory: {self._temp_dir}")

class GlassEngineFactory:
    @staticmethod
    def create(mode: GlassEngineMode, console: Optional[Console] = None) -> AbstractGlassEngine:
        from .modes.interactive_mode import InteractiveGlassEngine
        from .modes.demo_mode import DemoGlassEngine
        from .modes.debug_mode import DebugGlassEngine

        mode_map = {
            GlassEngineMode.INTERACTIVE: InteractiveGlassEngine,
            GlassEngineMode.DEMO: DemoGlassEngine,
            GlassEngineMode.DEBUG: DebugGlassEngine
        }
        if mode not in mode_map:
            raise ValueError(f"Unsupported Glass Engine mode: {mode}")
        return mode_map[mode](console=console)

async def run_glass_engine(mode: GlassEngineMode = GlassEngineMode.DEMO, console: Optional[Console] = None) -> GlassEngineMetrics:
    engine = GlassEngineFactory.create(mode, console)
    return await engine.run()
