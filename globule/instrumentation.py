"""OpenTelemetry instrumentation for Globule."""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult
)
from rich.console import Console
from rich.tree import Tree

class GlassEngineExporter(SpanExporter):
    """A custom OpenTelemetry span exporter for the Glass Engine."""
    def __init__(self, mode: str = 'surface'):
        self.mode = mode
        self.console = Console()

    def export(self, spans) -> SpanExportResult:
        if self.mode == 'surface':
            self._render_surface(spans)
        elif self.mode == 'mechanism':
            self._render_mechanism(spans)
        elif self.mode == 'core':
            self._render_core(spans)
        return SpanExportResult.SUCCESS

    def _render_surface(self, spans):
        for span in spans:
            self.console.print(f"[bold green]Trace:[/bold green] {span.name}")

    def _render_mechanism(self, spans):
        tree = Tree("[bold blue]Mechanism View[/bold blue]")
        for span in spans:
            duration = (span.end_time - span.start_time) / 1e6  # nanoseconds to milliseconds
            tree.add(f"{span.name} ([yellow]{duration:.2f}ms[/yellow])")
        self.console.print(tree)

    def _render_core(self, spans):
        for span in spans:
            tree = Tree(f"[bold red]Core View: {span.name}[/bold red]")
            duration = (span.end_time - span.start_time) / 1e6  # nanoseconds to milliseconds
            tree.add(f"[bold]Duration:[/bold] [yellow]{duration:.2f}ms[/yellow]")
            
            attributes_tree = tree.add("[bold]Attributes[/bold]")
            for key, value in span.attributes.items():
                attributes_tree.add(f"[cyan]{key}[/cyan]: {value}")

            events_tree = tree.add("[bold]Events[/bold]")
            for event in span.events:
                events_tree.add(f"[magenta]{event.name}[/magenta]")

            self.console.print(tree)

    def shutdown(self):
        pass

def setup_opentelemetry():
    """Configure OpenTelemetry for the application."""
    mode = os.getenv("GLASS_ENGINE_MODE", "surface")
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(GlassEngineExporter(mode=mode))
    )
