"""Interactive Mode for the Glass Engine.

A guided, step-by-step tutorial for new users.
"""

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode
from . import demo_scenes

class InteractiveGlassEngine(AbstractGlassEngine):
    """Implements the Interactive mode for the Glass Engine tutorial."""

    def get_mode(self) -> GlassEngineMode:
        return GlassEngineMode.INTERACTIVE

    async def execute_tutorial_flow(self) -> None:
        """Orchestrates the interactive tutorial, pausing for user input."""
        
        # Scene 1: Introduction
        demo_scenes.show_intro(self.console)

        # Scene 2: System Configuration
        self.console.input("\n[bold cyan]Press Enter to see the system configuration...[/bold cyan]")
        demo_scenes.show_system_configuration(self.console, self.api)

        # Scene 3: Capture Flow
        self.console.input("\n[bold cyan]Press Enter to see how a thought is captured...[/bold cyan]")
        await demo_scenes.demonstrate_capture_flow(self.console, self.api)

        # Scene 4: Retrieval Flow
        self.console.input("\n[bold cyan]Press Enter to see how thoughts are retrieved...[/bold cyan]")
        await demo_scenes.demonstrate_retrieval_flow(self.console, self.api)

        # Scene 5: Clustering Analysis (Board Impact!)
        self.console.input("\n[bold cyan]Press Enter to see AI-powered pattern discovery...[/bold cyan]")
        await demo_scenes.demonstrate_clustering_flow(self.console, self.api)

        # Scene 6: Final Summary
        self.console.input("\n[bold cyan]Press Enter to conclude the tutorial...[/bold cyan]")
        demo_scenes.show_final_summary(self.console)