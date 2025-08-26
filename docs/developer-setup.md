# Developer Setup Guide

Quick setup guide for developers who want to contribute to Globule or run it locally.

## Prerequisites

- **Python 3.8+** (3.11+ recommended)
- **Git** for version control
- **Ollama** (optional, for full AI functionality)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/globule
cd globule

# 2. Install in development mode
pip install -e .

# 3. Test the installation
globule tutorial --mode demo
```

## Full Development Setup

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e .

# Verify installation
globule --help
```

### 2. Ollama Setup (Optional but Recommended)

For full AI functionality, install Ollama:

```bash
# Install Ollama (see https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a small embedding model (in another terminal)
ollama pull nomic-embed-text

# Pull a small language model for parsing
ollama pull llama3.2:1b
```

**Without Ollama**: Globule gracefully degrades to mock providers for development.

### 3. Development Tools

```bash
# Install development dependencies (if you have them)
pip install pytest black flake8 mypy

# Run tests
pytest

# Code formatting
black src/

# Type checking
mypy src/globule/
```

## Project Structure

```
globule/
├── src/globule/          # Main package
│   ├── core/             # Core API and models
│   │   ├── api.py        # GlobuleAPI (main interface)
│   │   ├── models.py     # Pydantic data models
│   │   └── interfaces.py # Abstract base classes
│   ├── interfaces/       # Frontend implementations
│   │   ├── cli/          # Command-line interface
│   │   ├── tui/          # Terminal user interface
│   │   └── web/          # Web interface (placeholder)
│   ├── orchestration/    # Core business logic
│   │   └── engine.py     # OrchestrationEngine
│   ├── services/         # External service adapters
│   │   ├── embedding/    # Ollama embedding services
│   │   ├── parsing/      # Ollama parsing services
│   │   └── clustering/   # Semantic clustering
│   ├── storage/          # Data persistence layer
│   │   ├── sqlite_manager.py  # Main storage implementation
│   │   └── file_manager.py    # File system operations
│   └── tutorial/         # Glass Engine system
│       ├── glass_engine_core.py  # Base classes
│       └── modes/        # Tutorial mode implementations
├── docs/                 # Documentation
├── tests/                # Test suite
└── examples/             # Usage examples
```

## Architecture Overview

Globule follows a clean 3-layer architecture:

```
┌─────────────────────────────────────┐
│ Interface Layer                     │
│ (CLI, TUI, Web, Glass Engine)       │
├─────────────────────────────────────┤
│ API Layer                           │
│ (GlobuleAPI - core/api.py)          │
├─────────────────────────────────────┤
│ Core Logic Layer                    │
│ (Orchestration, Services, Storage)  │
└─────────────────────────────────────┘
```

**Key Principle**: All frontends use `GlobuleAPI` as the single source of truth. No direct service access from UI code.

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes following the architecture**:
   - UI changes: Modify interfaces, use GlobuleAPI only
   - Business logic: Modify core/orchestration/services
   - New API methods: Add to `core/api.py`

3. **Test your changes**:
   ```bash
   # Test basic functionality
   globule add "Test thought for development"
   globule search "test"
   
   # Test Glass Engine
   globule tutorial --mode debug
   ```

4. **Update documentation** if needed:
   - `docs/api.md` for new API methods
   - `docs/cli-reference.md` for new commands
   - `docs/architecture.md` for architectural changes

### Adding New Features

#### Adding a New API Method

```python
# In src/globule/core/api.py
async def new_feature(self, param: str) -> Any:
    """
    Description of new feature.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
    """
    # Use orchestrator or services, never direct imports
    result = await self.orchestrator.new_method(param)
    return result
```

#### Adding a New CLI Command

```python
# In src/globule/interfaces/cli/main.py
@click.command()
@click.argument('param', required=True)
@click.pass_context
async def new_command(ctx: click.Context, param: str) -> None:
    """Description of new command."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        result = await context.api.new_feature(param)
        click.echo(f"Result: {result}")

# Register the command
cli.add_command(new_command)
```

#### Adding a New Glass Engine Scene

```python
# In src/globule/tutorial/modes/demo_scenes.py
async def demonstrate_new_feature(console: Console, api: GlobuleAPI):
    """Demonstrate new feature in Glass Engine."""
    console.print(Panel.fit("[bold cyan]New Feature Demo[/bold cyan]"))
    
    with Progress(...) as progress:
        progress.add_task("Testing new feature...", total=None)
        result = await api.new_feature("demo_param")
    
    console.print(f"[bold green]Result: {result}[/bold green]")
```

## Testing

### Manual Testing

```bash
# Test core functionality
globule add "Test thought $(date)"
globule search "test"
globule cluster

# Test Glass Engine
globule tutorial --mode interactive
globule tutorial --mode debug

# Test advanced features
globule nlsearch "How many thoughts do I have?"
globule skeleton list
```

### Running Automated Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src/globule

# Run specific test file
pytest tests/unit/test_api.py
```

## Common Development Tasks

### Database Reset

```bash
# Remove existing database
rm ~/.globule/globule.db

# Or use a custom location
export GLOBULE_DB_PATH="/tmp/dev_globule.db"
```

### Working with Ollama

```bash
# Check Ollama status
ollama list

# Test embedding endpoint
curl http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"test"}'

# Pull different models
ollama pull llama3.2:3b
ollama pull mistral:7b
```

### Debug Mode

```bash
# Enable verbose logging
globule --verbose search "debug"

# Use Glass Engine debug mode for deep inspection
globule tutorial --mode debug
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'globule'**
```bash
# Ensure you're in the project directory and install in editable mode
pip install -e .
```

**Ollama Connection Error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

**Database Permission Error**
```bash
# Use a custom database location
export GLOBULE_DB_PATH="/tmp/globule_dev.db"
```

**Rich Console Display Issues**
```bash
# Ensure your terminal supports rich output
python -c "from rich.console import Console; Console().print('[bold]Test[/bold]')"
```

### Getting Help

1. **Check the documentation**: `docs/` directory
2. **Run Glass Engine debug mode**: `globule tutorial --mode debug`
3. **Use verbose flags**: `globule --verbose command`
4. **Check the API**: `docs/api.md`

## Contributing Guidelines

1. **Follow the architecture**: Always use GlobuleAPI from interfaces
2. **Write clean code**: Follow existing patterns and conventions
3. **Update documentation**: Keep docs in sync with code changes
4. **Test your changes**: Manual testing at minimum
5. **Use meaningful commit messages**: Explain what and why

## Next Steps

After setup, explore:

1. **Run the Glass Engine**: `globule tutorial --mode interactive`
2. **Read the architecture docs**: `docs/architecture.md`
3. **Study the API**: `docs/api.md`
4. **Try the CLI**: `docs/cli-reference.md`
5. **Examine the code**: Start with `src/globule/core/api.py`

Welcome to the Globule development community! 🎉