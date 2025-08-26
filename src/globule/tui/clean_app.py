"""
Clean TUI App: Pure presentation layer for Globule.

This is a complete rewrite of the TUI with ZERO business logic.
All data operations go through the GlobuleAPI. This is how a TUI should be built.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, TextArea, Input, Button
from textual.reactive import reactive
from textual.binding import Binding
from typing import List, Optional
import asyncio

from ..core.api import GlobuleAPI
from ..core.models import ProcessedGlobuleV1


class SearchResults(Static):
    """Widget to display search results."""
    
    def __init__(self, results: List[ProcessedGlobuleV1] = None):
        super().__init__()
        self.results = results or []
        self.update_display()
    
    def update_results(self, results: List[ProcessedGlobuleV1]):
        """Update the displayed results."""
        self.results = results
        self.update_display()
    
    def update_display(self):
        """Update the visual display of results."""
        if not self.results:
            self.update("No results found.")
            return
        
        content = []
        for i, globule in enumerate(self.results, 1):
            # Extract text from the globule structure
            text = getattr(globule, 'text', '') or str(globule.original_globule.raw_text)
            preview = text[:100] + "..." if len(text) > 100 else text
            content.append(f"{i}. {preview}")
        
        self.update("\n".join(content))


class DraftArea(TextArea):
    """Widget for composing drafts."""
    
    def __init__(self):
        super().__init__(placeholder="Draft content will appear here...")


class CleanGlobuleApp(App):
    """
    Clean, minimal TUI that delegates ALL business logic to GlobuleAPI.
    
    This is what the TUI should have been from the start:
    - Pure presentation layer
    - Zero direct database access
    - Zero direct API calls
    - Zero file I/O
    - Zero email sending
    - Just UI and API calls
    """
    
    CSS = """
    #search-container {
        height: 3;
        dock: top;
    }
    
    #main-container {
        layout: horizontal;
    }
    
    #left-panel {
        width: 50%;
        border-right: solid $accent;
    }
    
    #right-panel {
        width: 50%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+e", "export", "Export Draft"),
    ]
    
    def __init__(self, api: GlobuleAPI, initial_query: str = ""):
        super().__init__()
        self.api = api  # The ONLY way this TUI talks to the system
        self.initial_query = initial_query
        
        # UI components
        self.search_input: Optional[Input] = None
        self.search_results: Optional[SearchResults] = None
        self.draft_area: Optional[DraftArea] = None
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)
        
        with Vertical(id="search-container"):
            yield Input(
                placeholder="Enter search query...",
                value=self.initial_query,
                id="search-input"
            )
            yield Button("Search", id="search-button")
        
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield Static("Search Results:", classes="panel-title")
                yield SearchResults()
            
            with Vertical(id="right-panel"):
                yield Static("Draft:", classes="panel-title")
                yield DraftArea()
        
        yield Footer()
    
    def on_mount(self):
        """Initialize UI components after mount."""
        self.search_input = self.query_one("#search-input", Input)
        self.search_results = self.query_one(SearchResults)
        self.draft_area = self.query_one(DraftArea)
        
        # Perform initial search if query provided
        if self.initial_query:
            self.run_search(self.initial_query)
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "search-button":
            self.action_search()
    
    def on_input_submitted(self, event: Input.Submitted):
        """Handle input submission (Enter key)."""
        if event.input.id == "search-input":
            self.action_search()
    
    def action_search(self):
        """Trigger a search action."""
        if self.search_input:
            query = self.search_input.value.strip()
            if query:
                self.run_search(query)
    
    def run_search(self, query: str):
        """Execute search via API and update UI."""
        async def _search():
            try:
                # This is the ONLY business logic interaction - through the API
                results = await self.api.search_semantic(query, limit=20)
                self.search_results.update_results(results)
            except Exception as e:
                self.search_results.update_results([])
                self.notify(f"Search failed: {e}", severity="error")
        
        # Run the async search
        asyncio.create_task(_search())
    
    def action_export(self):
        """Export current draft."""
        async def _export():
            try:
                draft_content = self.draft_area.text
                if not draft_content.strip():
                    self.notify("Draft is empty", severity="warning")
                    return
                
                # Use API for export - NO direct file I/O
                success = await self.api.export_draft(
                    draft_content, 
                    f"draft_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                )
                
                if success:
                    self.notify("Draft exported successfully")
                else:
                    self.notify("Export failed", severity="error")
                    
            except Exception as e:
                self.notify(f"Export error: {e}", severity="error")
        
        asyncio.create_task(_export())
    
    def action_quit(self):
        """Quit the application."""
        self.exit()


def run_clean_tui(api: GlobuleAPI, query: str = "") -> None:
    """
    Run the clean TUI application.
    
    Args:
        api: The GlobuleAPI instance (dependency injection)
        query: Optional initial query to search
    """
    app = CleanGlobuleApp(api, query)
    app.run()