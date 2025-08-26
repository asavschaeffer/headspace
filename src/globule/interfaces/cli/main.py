"""
Main CLI commands for Globule.

Refactored to use the GlobuleAPI, providing a clean separation between the
command-line interface and the core application logic.
"""

import asyncio
import logging
import sys
from typing import Optional, Any

import asyncclick as click

from globule import __version__
from globule.core.api import GlobuleAPI
from globule.core.models import EnrichedInput, StructuredQuery
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.services.embedding.mock_adapter import MockEmbeddingAdapter as MockEmbeddingProvider
from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter
from globule.services.parsing.ollama_parser import OllamaParser
from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.orchestration.engine import GlobuleOrchestrator
from globule.core.frontend_manager import frontend_manager, FrontendType
from globule.config.settings import get_config
from globule.schemas.manager import SchemaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobuleContext:
    """Shared context for CLI commands, centered around the GlobuleAPI."""

    def __init__(self):
        self.api: Optional[GlobuleAPI] = None
        self._storage = None
        self._embedding_provider = None
        self._parsing_provider = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        await self.cleanup()

    async def initialize(self, verbose: bool = False) -> None:
        """Initialize all services and create the GlobuleAPI instance."""
        if self._initialized:
            return

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config = get_config()

        # 1. Initialize Storage
        self._storage = SQLiteStorageManager()
        await self._storage.initialize()

        # 2. Initialize Embedding Provider
        embedding_provider = OllamaEmbeddingProvider()
        if not await embedding_provider.health_check():
            click.echo("Warning: Ollama not accessible. Using mock embeddings.", err=True)
            await embedding_provider.close()
            embedding_provider = MockEmbeddingProvider()
        self._embedding_provider = embedding_provider
        embedding_adapter = OllamaEmbeddingAdapter(self._embedding_provider)

        # 3. Initialize Parsing Provider
        self._parsing_provider = OllamaParser()
        parsing_adapter = OllamaParsingAdapter(self._parsing_provider)

        # 4. Initialize Orchestrator
        orchestrator = GlobuleOrchestrator(
            embedding_provider=embedding_adapter,
            parser_provider=parsing_adapter,
            storage_manager=self._storage
        )

        # 5. Create the API
        self.api = GlobuleAPI(storage=self._storage, orchestrator=orchestrator)

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up all services."""
        if not self._initialized:
            return

        if self._embedding_provider:
            await self._embedding_provider.close()
        if self._parsing_provider:
            await self._parsing_provider.close()
        if self._storage:
            await self._storage.close()
        self._initialized = False


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
async def cli(ctx: click.Context, verbose: bool):
    """
    Globule: Turn your scattered thoughts into structured drafts. Effortlessly.
    """
    ctx.ensure_object(dict)
    ctx.obj['context'] = GlobuleContext()
    ctx.obj['verbose'] = verbose


@click.command()
@click.argument('text', required=True)
@click.pass_context
async def add(ctx: click.Context, text: str) -> None:
    """Add a thought to your Globule collection."""
    verbose = ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            click.echo("Processing your thought...")
            processed_globule = await context.api.add_thought(text, source="cli")
            click.echo(f"Thought captured with ID: {processed_globule.globule_id}")

        except Exception as e:
            logger.error(f"Failed to add thought: {e}")
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


@click.command()
@click.argument('topic', required=True)
@click.option('--limit', '-l', default=100, help='Maximum globules to search.')
@click.option('--output', '-o', help='Output draft to file.')
@click.option('--frontend', '-f', type=click.Choice(['tui', 'web', 'cli'], case_sensitive=False),
              default='tui', help='Frontend to use.')
@click.option('--port', '-p', default=8000, help='Port for web frontend.')
@click.option('--host', default='127.0.0.1', help='Host for web frontend.')
@click.pass_context
async def draft(ctx: click.Context, topic: str, limit: int, output: Optional[str],
                frontend: str, port: int, host: str) -> None:
    """Interactive drafting from clustered thoughts."""
    verbose = ctx.obj.get('verbose', False)
    frontend_type = FrontendType(frontend.lower())
    
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)

        if frontend_type == FrontendType.CLI:
            click.echo("[CLI] Searching for related thoughts...")
            results = await context.api.search_semantic(topic, limit=limit)
            if not results:
                click.echo("No related thoughts found.")
                return

            click.echo(f"Found {len(results)} related thoughts.")
            draft_content = "\n\n---\n\n".join([g.original_globule.raw_text for g in results])

            if output:
                if await context.api.export_draft(draft_content, output):
                    click.echo(f"Draft exported to: {output}")
                else:
                    click.echo(f"Error exporting draft to: {output}", err=True)
            else:
                click.echo("\n--- Draft Content ---\
")
                click.echo(draft_content)
            return

        launch_kwargs = {
            'topic': topic,
            'limit': limit,
            'output': output,
            'api': context.api
        }
        if frontend_type == FrontendType.WEB:
            launch_kwargs.update({'port': port, 'host': host})

        result = await frontend_manager.launch_frontend(frontend_type, **launch_kwargs)

        if not result['success']:
            click.echo(f"[ERROR] {result['message']}", err=True)
            raise click.Abort()

        click.echo(f"[SUCCESS] {result['message']}")


@click.command()
@click.argument('query', required=True)
@click.option('--limit', '-l', default=10, help='Maximum results to return.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed search results.')
@click.pass_context
async def search(ctx: click.Context, query: str, limit: int, verbose: bool) -> None:
    """Search for similar thoughts using semantic vector search."""
    verbose = verbose or ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo(f"Searching for: '{query}'")
        results = await context.api.search_semantic(query, limit=limit)

        if not results:
            click.echo("No similar thoughts found.")
            return

        click.echo(f"Found {len(results)} similar thoughts:\n")
        for i, globule in enumerate(results, 1):
            click.echo(f"{i}. {globule.original_globule.raw_text[:100]}...")
            if verbose:
                click.echo(f"   ID: {globule.globule_id}")
                click.echo(f"   Created: {globule.processed_timestamp}")

@click.command()
@click.pass_context
async def reconcile(ctx: click.Context) -> None:
    """Reconcile files on disk with the database."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        click.echo("Starting file-database reconciliation...")
        stats = await context.api.reconcile_files()
        click.echo("Reconciliation Complete:")
        for key, value in stats.items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")

@click.command()
@click.argument('query', required=True)
@click.pass_context
async def nlsearch(ctx: click.Context, query: str) -> None:
    """Ask a natural language question about your thoughts."""
    verbose = ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo(f"Answering question: '{query}'...")
        try:
            results = await context.api.natural_language_query(query)
            if not results:
                click.echo("Could not answer the question.")
                return

            from rich.table import Table
            from rich.console import Console
            table = Table(title=f"Result for: '{query}'")
            headers = results[0].keys() if results else []
            for header in headers:
                table.add_column(header, justify="left")
            for row in results:
                table.add_row(*[str(item) for item in row.values()])
            console = Console()
            console.print(table)

        except Exception as e:
            click.echo(f"Error during natural language query: {e}", err=True)

# Tutorial command removed - tutorials are now in examples/ directory
# Run tutorials with: python examples/tutorials/glass_engine_core.py

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed cluster analysis')
@click.option('--export', '-e', help='Export results to JSON file')
@click.pass_context
async def cluster(ctx: click.Context, verbose: bool, export: Optional[str]) -> None:
    """Discover semantic clusters and themes in your thoughts."""
    verbose = verbose or ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo("Analyzing semantic clusters in your thoughts...")
        analysis = await context.api.get_clusters()
        if not analysis.clusters:
            click.echo("No clusters found. Add more thoughts to enable clustering.")
            return

        click.echo(f"\nDiscovered {len(analysis.clusters)} semantic clusters:\n")
        for i, cluster_obj in enumerate(analysis.clusters, 1):
            click.echo(f"{i}. {cluster_obj.label} ({cluster_obj.size} thoughts)")
            if verbose:
                click.echo(f"   Description: {cluster_obj.description}")
                click.echo(f"   Keywords: {', '.join(cluster_obj.keywords)}")
                click.echo(f"   Confidence: {cluster_obj.confidence_score:.2f}")
        
        if export:
            import json
            try:
                with open(export, 'w') as f:
                    json.dump(analysis.to_dict(), f, indent=2)
                click.echo(f"\nExported analysis to {export}")
            except Exception as e:
                click.echo(f"\nError exporting to file: {e}", err=True)

@click.group()
def skeleton():
    """Manage canvas skeleton templates."""
    pass

@skeleton.command(name="list")
@click.pass_context
async def skeleton_list(ctx: click.Context):
    """List all available skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        skeletons = context.api.list_skeletons()
        if not skeletons:
            click.echo("No skeletons found.")
            return
        click.echo("Available Skeletons:")
        for s in skeletons:
            click.echo(f"- {s['name']}: {s['description']}")

@skeleton.command(name="apply")
@click.argument('name', required=True)
@click.pass_context
async def skeleton_apply(ctx: click.Context, name: str):
    """Apply a skeleton template."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        click.echo(f"Applying skeleton: {name}...")
        modules = context.api.apply_skeleton(name)
        click.echo("Generated Modules:")
        for m in modules:
            click.echo(f"- {m['name']}")

@skeleton.command(name="stats")
@click.pass_context
async def skeleton_stats(ctx: click.Context):
    """Show statistics about skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        stats = context.api.get_skeleton_stats()
        click.echo("Skeleton Stats:")
        for key, value in stats.items():
            click.echo(f"- {key.replace('_', ' ').title()}: {value}")

@skeleton.command(name="create-defaults")
@click.pass_context
async def skeleton_create_defaults(ctx: click.Context):
    """Create default skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        created = context.api.create_default_skeletons()
        click.echo(f"Created {len(created)} default skeletons: {', '.join(created)}")

@click.group()
def inputs():
    """Manage external input sources (WhatsApp, email, etc.)."""
    pass

@inputs.command(name="setup-whatsapp")
@click.option('--access-token', required=True, help='WhatsApp Business API access token')
@click.option('--verify-token', required=True, help='Webhook verification token')
@click.option('--phone', multiple=True, help='Authorized phone numbers (can specify multiple)')
@click.option('--relay-url', help='Relay service URL (e.g., https://relay.globule.app)')
@click.pass_context
async def setup_whatsapp(ctx: click.Context, access_token: str, verify_token: str, phone: tuple, relay_url: str):
    """Set up WhatsApp as an input source."""
    try:
        from globule.inputs.manager import InputSourceManager
        
        # Create input manager and register WhatsApp
        input_manager = InputSourceManager()
        input_manager.register_whatsapp(access_token, verify_token, list(phone) if phone else None)
        
        click.echo(f"✅ WhatsApp input source configured")
        if phone:
            click.echo(f"📱 Authorized phone numbers: {', '.join(phone)}")
        
        if relay_url:
            click.echo(f"\n📡 To complete setup, register with relay service:")
            click.echo(f"POST {relay_url}/register")
            click.echo(f"{{")
            click.echo(f'  "user_id": "your_user_id",')
            click.echo(f'  "endpoint_url": "https://your-ngrok-url.ngrok.io/webhook",')
            click.echo(f'  "auth_token": "your_secret_token",')
            click.echo(f'  "platforms": {{')
            if phone:
                for p in phone:
                    click.echo(f'    "whatsapp": "{p}"')
            else:
                click.echo(f'    "whatsapp": "+1234567890"')
            click.echo(f'  }}')
            click.echo(f'}}')
        else:
            click.echo("\n💡 Tip: Use --relay-url to get setup instructions for the relay service")
            
    except ImportError as e:
        click.echo(f"❌ Failed to import inputs module: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Failed to setup WhatsApp: {e}", err=True)

@inputs.command(name="setup-telegram")
@click.option('--bot-token', required=True, help='Telegram bot token from @BotFather')
@click.option('--user-id', multiple=True, type=int, help='Authorized Telegram user IDs (can specify multiple)')
@click.option('--relay-url', help='Relay service URL (e.g., https://relay.globule.app)')
@click.pass_context
async def setup_telegram(ctx: click.Context, bot_token: str, user_id: tuple, relay_url: str):
    """Set up Telegram as an input source."""
    try:
        from globule.inputs.manager import InputSourceManager
        
        # Create input manager and register Telegram
        input_manager = InputSourceManager()
        input_manager.register_telegram(bot_token, list(user_id) if user_id else None)
        
        click.echo(f"✅ Telegram input source configured")
        if user_id:
            click.echo(f"👤 Authorized user IDs: {', '.join(map(str, user_id))}")
        
        if relay_url:
            click.echo(f"\n📡 To complete setup, register with relay service:")
            click.echo(f"POST {relay_url}/register")
            click.echo(f"And set webhook URL to: {relay_url}/webhook/telegram/{bot_token}")
        else:
            click.echo("\n💡 Tip: Use --relay-url to get setup instructions for the relay service")
            
    except ImportError as e:
        click.echo(f"❌ Failed to import inputs module: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Failed to setup Telegram: {e}", err=True)

@inputs.command(name="webhook-server")
@click.option('--port', default=8080, help='Port to run webhook server on')
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--tunnel', is_flag=True, help='Auto-create ngrok tunnel for public access')
@click.pass_context
async def webhook_server(ctx: click.Context, port: int, host: str, tunnel: bool):
    """Run a local webhook server to receive messages from relay service."""
    try:
        from aiohttp import web
        from globule.inputs.manager import InputSourceManager
        
        click.echo(f"🚀 Starting webhook server on {host}:{port}")
        
        if tunnel:
            click.echo("🌐 Creating public tunnel with ngrok...")
            # TODO: Add ngrok integration
            click.echo("💡 Install ngrok and run: ngrok http " + str(port))
        
        # Create webhook handler
        input_manager = InputSourceManager()
        
        async def webhook_handler(request):
            """Handle incoming webhook messages."""
            try:
                data = await request.json()
                platform = data.get('platform')
                payload = data.get('payload', {})
                
                # Process based on platform
                if platform == 'whatsapp':
                    messages = await input_manager.process_whatsapp_webhook(payload)
                    for msg in messages:
                        # Process through API
                        async with ctx.obj['context'] as context:
                            await context.initialize()
                            results = await context.api.add_from_input_message(msg)
                            click.echo(f"📨 Processed WhatsApp message: {len(results)} globules created")
                
                elif platform == 'telegram':
                    message = await input_manager.process_telegram_webhook(payload)
                    if message:
                        async with ctx.obj['context'] as context:
                            await context.initialize()
                            results = await context.api.add_from_input_message(message)
                            click.echo(f"📨 Processed Telegram message: {len(results)} globules created")
                
                return web.json_response({"status": "processed"})
                
            except Exception as e:
                click.echo(f"❌ Webhook processing failed: {e}", err=True)
                return web.json_response({"error": str(e)}, status=500)
        
        # Create web app
        app = web.Application()
        app.router.add_post('/webhook', webhook_handler)
        app.router.add_get('/health', lambda r: web.json_response({"status": "healthy"}))
        
        # Run server
        web.run_app(app, host=host, port=port)
        
    except ImportError as e:
        click.echo(f"❌ Failed to import required modules: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Failed to start webhook server: {e}", err=True)

@inputs.command(name="test-message")
@click.option('--source', default='test', help='Source identifier')
@click.option('--content', required=True, help='Test message content')
@click.pass_context  
async def test_message(ctx: click.Context, source: str, content: str):
    """Test message processing with a mock InputMessage."""
    try:
        from globule.inputs.models import InputMessage
        
        # Create test message
        test_msg = InputMessage(
            content=content,
            source=source,
            user_identifier="test_user"
        )
        
        async with ctx.obj['context'] as context:
            await context.initialize()
            results = await context.api.add_from_input_message(test_msg)
            
            click.echo(f"✅ Test message processed successfully!")
            click.echo(f"📊 Created {len(results)} globules:")
            for i, globule in enumerate(results, 1):
                preview = globule.original_globule.raw_text[:50] + "..." if len(globule.original_globule.raw_text) > 50 else globule.original_globule.raw_text
                click.echo(f"  {i}. {preview}")
        
    except ImportError as e:
        click.echo(f"❌ Failed to import inputs module: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Test message processing failed: {e}", err=True)

@click.command()
@click.argument('path', required=True)
@click.option('--include', '-i', multiple=True, help='Include patterns (e.g., *.md, *.jpg). Processes all supported content if not specified.')
@click.option('--exclude', '-e', multiple=True, help='Exclude patterns (e.g., *.tmp, *.log). Adds to default exclusions.')
@click.option('--dry-run', is_flag=True, help='Show what would be indexed without actually indexing')
@click.pass_context
async def index(ctx: click.Context, path: str, include: tuple, exclude: tuple, dry_run: bool) -> None:
    """
    Index all processable content in a directory for read-only analysis (Phase A - Index-First).
    
    This command intelligently processes all supported content types (text, images, PDFs, etc.)
    and stores them in the database without modifying or moving any files. This is the safe,
    read-only entry point to Globule that leverages its multi-modal processing capabilities.
    
    Examples:
        globule index ~/Documents/notes
        globule index /path/to/project --include "*.md" "*.jpg"
        globule index . --exclude "*.tmp" --dry-run
    """
    verbose = ctx.obj.get('verbose', False)
    
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            
            # Convert path to absolute path
            from pathlib import Path
            absolute_path = Path(path).resolve()
            
            if not absolute_path.exists():
                click.echo(f"❌ Path does not exist: {absolute_path}", err=True)
                raise click.Abort()
            
            if not absolute_path.is_dir():
                click.echo(f"❌ Path is not a directory: {absolute_path}", err=True)
                raise click.Abort()
            
            click.echo(f"🔍 {'[DRY RUN] ' if dry_run else ''}Indexing processable content in: {absolute_path}")
            if include:
                click.echo(f"📂 Including: {', '.join(include)}")
            if exclude:
                click.echo(f"🚫 Excluding: {', '.join(exclude)} (plus defaults)")
            
            if dry_run:
                # For dry run, estimate files that would be processed
                import os
                import fnmatch
                
                # Use same default exclusions as the API
                default_excludes = ['*.tmp', '*.log', '*.cache', '.DS_Store', 'thumbs.db', '*.lock']
                all_excludes = list(exclude) + default_excludes
                
                count = 0
                for root, dirs, files in os.walk(absolute_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git']]
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        # Apply include/exclude patterns
                        if include and not any(fnmatch.fnmatch(file, pattern) for pattern in include):
                            continue
                            
                        if any(fnmatch.fnmatch(file, pattern) for pattern in all_excludes):
                            continue
                            
                        count += 1
                
                click.echo(f"📊 Would attempt to process {count} files")
                click.echo(f"💡 Globule's processor router will determine what can actually be indexed")
                click.echo(f"🎯 Supports: text files, images, PDFs, and more via intelligent content detection")
                return
            
            # Perform actual indexing
            include_patterns = list(include) if include else None
            exclude_patterns = list(exclude) if exclude else None
            stats = await context.api.index_path(str(absolute_path), 
                                                 include_patterns=include_patterns,
                                                 exclude_patterns=exclude_patterns)
            
            # Display results
            click.echo(f"\n✅ Indexing completed!")
            click.echo(f"📁 Files processed: {stats['files_processed']}")
            click.echo(f"📝 Files indexed: {stats['files_indexed']}")
            click.echo(f"⏭️  Files skipped: {stats['files_skipped']}")
            click.echo(f"🔄 Files unsupported: {stats.get('files_unsupported', 0)}")
            click.echo(f"❌ Files failed: {stats['files_failed']}")
            click.echo(f"📊 Total size: {stats['total_size_bytes']:,} bytes")
            
            # Show content types discovered
            if stats.get('content_types'):
                click.echo(f"\n📋 Content types indexed:")
                for content_type, count in stats['content_types'].items():
                    click.echo(f"   • {content_type}: {count} files")
            
            if stats['errors']:
                click.echo(f"\n⚠️  Errors encountered:")
                for error in stats['errors'][:5]:  # Show first 5 errors
                    click.echo(f"   {error}")
                if len(stats['errors']) > 5:
                    click.echo(f"   ... and {len(stats['errors']) - 5} more errors")
            
            if stats['files_indexed'] > 0:
                click.echo(f"\n💡 Next steps:")
                click.echo(f"   • Search your indexed content: globule search '<query>'")
                click.echo(f"   • Discover themes: globule cluster")
                click.echo(f"   • Organize into files: globule organize --dry-run")
                
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            click.echo(f"❌ Indexing failed: {e}", err=True)
            raise click.Abort()

@click.command()
@click.option('--output-dir', '-o', help='Directory where organized files should be created')
@click.option('--dry-run', is_flag=True, help='Show proposed organization without creating files')
@click.option('--interactive', '-i', is_flag=True, help='Interactively approve organization structure')
@click.pass_context
async def organize(ctx: click.Context, output_dir: str, dry_run: bool, interactive: bool) -> None:
    """
    Organize indexed content into structured files (Phase B - Index-First).
    
    This command takes your indexed content and creates organized markdown files
    with AI-generated directory structures based on semantic clustering.
    
    Examples:
        globule organize --dry-run
        globule organize --output-dir ~/organized-notes
        globule organize --interactive
    """
    verbose = ctx.obj.get('verbose', False)
    
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            
            click.echo("📋 Analyzing indexed content for organization...")
            
            # Always do dry run first to show proposed structure
            results = await context.api.organize_repository(output_dir=output_dir, dry_run=True)
            
            if results.get("status") == "no_unmanaged_globules":
                click.echo("📭 No unmanaged content found to organize.")
                click.echo("💡 Tip: Use 'globule index <path>' to index content first.")
                return
            
            # Display analysis results
            click.echo(f"📊 Analysis Results:")
            click.echo(f"   • Unmanaged files: {results['unmanaged_count']}")
            click.echo(f"   • Semantic clusters: {results['clusters_found']}")
            
            structure = results.get('proposed_structure', {})
            if structure.get('type') == 'clustered':
                click.echo(f"\n📁 Proposed Directory Structure:")
                for dir_name, info in structure.get('directories', {}).items():
                    click.echo(f"   📂 {dir_name}/ ({info['globule_count']} files)")
                    click.echo(f"      {info['description']}")
                    click.echo(f"      Keywords: {', '.join(info['keywords'][:5])}")
            elif structure.get('type') == 'flat':
                click.echo(f"\n📁 Content will be organized into: {structure.get('directories', ['unsorted'])}")
            
            if dry_run:
                click.echo(f"\n💡 This was a dry run. Use without --dry-run to create files.")
                return
            
            if interactive:
                if not click.confirm("\nDo you want to proceed with this organization?"):
                    click.echo("Organization cancelled.")
                    return
            
            # TODO: Implement actual file creation (not yet implemented)
            click.echo(f"\n⚠️  File creation not yet implemented.")
            click.echo(f"The organize command currently supports analysis and dry-run only.")
            click.echo(f"Full file creation will be implemented in the next phase.")
                
        except Exception as e:
            logger.error(f"Organization failed: {e}")
            click.echo(f"❌ Organization failed: {e}", err=True)
            raise click.Abort()

# Register all commands
cli.add_command(add)
cli.add_command(draft)
cli.add_command(search)
cli.add_command(reconcile)
cli.add_command(cluster)
cli.add_command(nlsearch)
cli.add_command(skeleton)
cli.add_command(inputs)
cli.add_command(index)
cli.add_command(organize)

def main():
    """Entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()