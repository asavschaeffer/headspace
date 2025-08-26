"""Demo Mode for the Glass Engine."""

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode
from . import demo_scenes

class DemoGlassEngine(AbstractGlassEngine):
    """Implements the Demo mode for the Glass Engine tutorial."""

    def get_mode(self) -> GlassEngineMode:
        return GlassEngineMode.DEMO

    async def execute_tutorial_flow(self) -> None:
        """Orchestrates the demo by calling scenes in order."""
        
        # Scene 1: Introduction
        demo_scenes.show_intro(self.console)

        # Scene 2: System Configuration
        demo_scenes.show_system_configuration(self.console, self.api)

        # Scene 3: Capture Flow
        await demo_scenes.demonstrate_capture_flow(self.console, self.api)

        # Scene 4: Retrieval Flow
        await demo_scenes.demonstrate_retrieval_flow(self.console, self.api)

        # Scene 5: Final Summary
        demo_scenes.show_final_summary(self.console)
