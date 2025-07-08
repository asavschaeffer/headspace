"""CLI interface for Globule."""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel

from .config import load_config, create_default_config
from .processor import process_thought, cleanup
from .storage import SQLiteStorage
from .embedding_engine import create_embedder
from .query_engine import QueryEngine, parse_query
from .synthesis import SynthesisEngine
from .instrumentation import setup_opentelemetry
import os

console = Console()

class AppContext:
    """A context object to hold shared resources."""
    def __init__(self):
        try:
            self.config = load_config()
            self.storage = SQLiteStorage(self.config.db_path)
        except Exception:
            # Config command will handle creation
            self.config = None
            self.storage = None

@click.group()
@click.version_option()
@click.pass_context
def cli(ctx: click.Context):
    """Globule: A semantic thought processor for capturing and retrieving thoughts intelligently."""
    if os.getenv("GLASS_ENGINE_ENABLED", "false").lower() == "true":
        setup_opentelemetry()
    ctx.obj = AppContext()

@cli.command()
@click.argument('text')
def add(text: str):
    """Add a new thought to your globule collection."""
    console.print(Panel(f"✓ Captured: {text}", title="Processing", border_style="green"))
    console.print("[dim]Processing in background...[/dim]")
    
    async def process():
        try:
            globule_id = await process_thought(text)
            console.print(f"[green]✓ Thought captured successfully![/green]")
            console.print(f"[dim]ID: {globule_id}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            await cleanup()
    
    asyncio.run(process())

@cli.command()
@click.argument('query')
@click.pass_context
def search(ctx: click.Context, query: str):
    """Search for thoughts using semantic similarity."""
    console.print(Panel(f"🔍 Searching for: {query}", title="Search", border_style="blue"))
    
    async def search_thoughts():
        try:
            config = ctx.obj.config
            storage = ctx.obj.storage
            embedder = await create_embedder(use_ollama=config.embedding_provider == "local")
            query_engine = QueryEngine(storage, embedder)
            
            # Parse the query for additional parameters
            query_params = parse_query(query)
            
            # Perform search
            results = await query_engine.search_combined(
                query_params["clean_query"],
                domain=query_params.get("domain"),
                timeframe=query_params.get("timeframe"),
                limit=10
            )
            
            query_engine.display_results(results, f"Results for '{query}'")
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
    
    asyncio.run(search_thoughts())

@cli.command()
@click.pass_context
def today(ctx: click.Context):
    """Show today's thoughts."""
    console.print(Panel("📅 Today's Thoughts", title="Daily View", border_style="cyan"))
    
    async def show_today():
        try:
            config = ctx.obj.config
            storage = ctx.obj.storage
            embedder = await create_embedder(use_ollama=config.embedding_provider == "local")
            query_engine = QueryEngine(storage, embedder)
            
            today_globules = await query_engine.search_temporal("today")
            query_engine.display_temporal_results(today_globules, "Today's Thoughts")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(show_today())

@cli.command()
@click.pass_context
def report(ctx: click.Context):
    """Generate a daily summary report."""
    console.print(Panel("📊 Daily Summary", title="Report", border_style="magenta"))
    
    async def generate_report():
        try:
            config = ctx.obj.config
            storage = ctx.obj.storage
            embedder = await create_embedder(use_ollama=config.embedding_provider == "local")
            query_engine = QueryEngine(storage, embedder)
            synthesis_engine = SynthesisEngine(query_engine, use_llm=config.llm_provider == "local")
            
            await synthesis_engine.initialize()
            
            summary = await synthesis_engine.generate_daily_summary()
            synthesis_engine.display_summary(summary)
            
        except Exception as e:
            console.print(f"[red]Report error: {e}[/red]")
    
    asyncio.run(generate_report())

@cli.command()
@click.pass_context
def stats(ctx: click.Context):
    """Show database statistics."""
    console.print(Panel("📈 Database Statistics", title="Stats", border_style="yellow"))
    
    async def show_stats():
        try:
            config = ctx.obj.config
            storage = ctx.obj.storage
            embedder = await create_embedder(use_ollama=config.embedding_provider == "local")
            query_engine = QueryEngine(storage, embedder)
            
            stats = await query_engine.get_stats()
            query_engine.display_stats(stats)
            
        except Exception as e:
            console.print(f"[red]Stats error: {e}[/red]")
    
    asyncio.run(show_stats())

@cli.command()
@click.pass_context
def config(ctx: click.Context):
    """Show or create configuration."""
    app_config = ctx.obj.config
    if app_config:
        console.print(Panel("Current Configuration", border_style="blue"))
        console.print(f"LLM Provider: {app_config.llm_provider}")
        console.print(f"LLM Model: {app_config.llm_model}")
        console.print(f"Embedding Provider: {app_config.embedding_provider}")
        console.print(f"Embedding Model: {app_config.embedding_model}")
        console.print(f"Database Path: {app_config.db_path}")
        console.print(f"Report Template: {app_config.report_template}")
    else:
        console.print("[yellow]No config found, creating default...[/yellow]")
        create_default_config()
        console.print("[green]Created config.yaml[/green]")

if __name__ == '__main__':
    cli()
