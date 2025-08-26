"""Debug Mode for the Glass Engine.

A mode for developers to see raw data structures and detailed logs.
"""

import logging
import time
from rich import print

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode

class DebugGlassEngine(AbstractGlassEngine):
    """Implements the Debug mode for the Glass Engine tutorial."""

    def get_mode(self) -> GlassEngineMode:
        return GlassEngineMode.DEBUG

    async def execute_tutorial_flow(self) -> None:
        """Performs all core API actions and prints the raw results."""
        self.logger.setLevel(logging.DEBUG)
        self.console.print("[bold red]>>> RUNNING IN DEBUG MODE <<<", justify="center")

        # --- Add Thought ---
        self.console.print("\n[bold cyan]1. Testing api.add_thought(...)[/bold cyan]")
        test_thought = "Debug mode test: Check data structures."
        self.console.print(f"Input: \"{test_thought}\"")
        start_time = time.time()
        added_globule = await self.api.add_thought(test_thought, source="glass_engine_debug")
        duration = (time.time() - start_time) * 1000
        self.console.print(f"[dim]Performance: {duration:.1f}ms[/dim]")
        self.console.print("[bold green]Output:[/bold green]")
        print(added_globule.dict())

        # --- Semantic Search ---
        self.console.print("\n[bold cyan]2. Testing api.search_semantic(...)[/bold cyan]")
        test_query = "data structures"
        self.console.print(f"Input: \"{test_query}\"")
        start_time = time.time()
        search_results = await self.api.search_semantic(test_query, limit=2)
        duration = (time.time() - start_time) * 1000
        self.console.print(f"[dim]Performance: {duration:.1f}ms[/dim]")
        self.console.print("[bold green]Output:[/bold green]")
        print([g.dict() for g in search_results])

        # --- Clustering ---
        self.console.print("\n[bold cyan]3. Testing api.get_clusters(...)[/bold cyan]")
        start_time = time.time()
        clustering_analysis = await self.api.get_clusters()
        duration = (time.time() - start_time) * 1000
        self.console.print(f"[dim]Performance: {duration:.1f}ms[/dim]")
        self.console.print("[bold green]Output:[/bold green]")
        print(clustering_analysis.to_dict())

        # --- NL Search ---
        self.console.print("\n[bold cyan]4. Testing api.natural_language_query(...)[/bold cyan]")
        nl_query = "how many globules are there"
        self.console.print(f"Input: \"{nl_query}\"")
        start_time = time.time()
        nl_results = await self.api.natural_language_query(nl_query)
        duration = (time.time() - start_time) * 1000
        self.console.print(f"[dim]Performance: {duration:.1f}ms[/dim]")
        self.console.print("[bold green]Output:[/bold green]")
        print(nl_results)

        self.console.print("\n[bold red]>>> DEBUG MODE COMPLETE <<<", justify="center")