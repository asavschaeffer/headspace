# Glass Engine Guide

The Glass Engine is Globule's signature transparency system - a unified tutorial, testing, and showcase platform that demonstrates how the application works from the inside out.

## Philosophy

Most software is a "black box" - you input something and hope for the best. The Glass Engine makes Globule completely transparent, allowing users to see exactly how their thoughts flow through the system. This serves multiple purposes:

1. **Education**: New users understand what the system does
2. **Validation**: Every tutorial run is also a system test
3. **Showcase**: Demonstrates capabilities to stakeholders
4. **Debugging**: Developers can inspect system internals

## Three Modes for Different Audiences

The Glass Engine provides three distinct modes, each tailored for specific use cases:

### 🎯 Demo Mode (Professional Showcase)
**Target Audience**: Stakeholders, potential users, marketing demos
**Use Case**: Quick, professional demonstration of core capabilities

- **Duration**: 2-3 minutes
- **Format**: Automated presentation with rich visual formatting
- **Content**: Shows thought capture → processing → retrieval workflow
- **Output**: Beautiful rich console output with panels and progress bars

```bash
globule tutorial --mode demo
```

### 🎓 Interactive Mode (Board Meeting Ready)
**Target Audience**: Executives, investors, technical presentations
**Use Case**: Guided walkthrough emphasizing AI capabilities and business value

- **Duration**: 5-10 minutes with user pacing
- **Format**: Step-by-step tutorial with manual progression
- **Content**: 
  - System configuration transparency
  - Live thought capture demonstration
  - Semantic search capabilities
  - AI-powered pattern discovery (clustering)
  - Professional result formatting
- **Output**: Impressive visual presentation with user control

```bash
globule tutorial --mode interactive
```

**Interactive Mode Flow:**
1. **Introduction**: Welcome and mission statement
2. **Configuration**: Show system architecture transparency
3. **Capture Flow**: Demonstrate `api.add_thought()` with live processing
4. **Retrieval Flow**: Show `api.search_semantic()` with formatted results
5. **Pattern Discovery**: Showcase `api.get_clusters()` for AI capabilities
6. **Summary**: Highlight the complete transparency achieved

### 🔧 Debug Mode (Developer Deep Dive)
**Target Audience**: Developers, system administrators, technical debugging
**Use Case**: Raw system inspection and performance analysis

- **Duration**: Variable, depends on investigation needs
- **Format**: Raw data dumps with performance metrics
- **Content**:
  - All API method testing (`add_thought`, `search_semantic`, `get_clusters`, `natural_language_query`)
  - Raw object inspection with `.dict()` output
  - Performance timing for each operation
  - Complete system state visibility
- **Output**: Unformatted technical data for analysis

```bash
globule tutorial --mode debug
```

**Debug Mode Output Example:**
```
>>> RUNNING IN DEBUG MODE <<<

1. Testing api.add_thought(...)
Input: "Debug mode test: Check data structures."
Performance: 1247.3ms
Output:
{'globule_id': '123e4567-e89b-12d3-a456-426614174000', 'embedding': [0.1, 0.2, ...], ...}

2. Testing api.search_semantic(...)
Input: "data structures"  
Performance: 432.1ms
Output:
[{'globule_id': '...', 'original_globule': {'raw_text': '...'}, ...}]
```

## Architecture Integration

The Glass Engine exemplifies Globule's clean architecture:

```
Glass Engine (Interface Layer)
     ↓
GlobuleAPI (API Layer)  
     ↓
OrchestrationEngine + Services (Core Logic Layer)
```

**Key Design Principles:**
- **API Client**: Glass Engine acts as a client of GlobuleAPI, never accessing services directly
- **Sandboxed Environment**: Creates isolated temporary database for safe experimentation
- **Zero Business Logic**: Pure presentation layer with all logic delegated to API
- **Metrics Collection**: Tracks performance and validation status
- **Resource Management**: Proper cleanup of temporary resources

## Technical Implementation

### Core Components

**`glass_engine_core.py`**: Abstract base class and infrastructure
- `AbstractGlassEngine`: Base class for all modes
- `GlassEngineMetrics`: Performance and validation tracking
- Sandboxed environment creation and cleanup
- GlobuleAPI initialization and dependency injection

**`modes/`**: Individual mode implementations
- `demo_mode.py`: Professional automated showcase
- `interactive_mode.py`: Guided user-paced tutorial
- `debug_mode.py`: Raw technical inspection
- `demo_scenes.py`: Reusable presentation components

### Sandboxed Environment

Each Glass Engine run creates a completely isolated environment:

```python
# Temporary directory for database and files
temp_dir = tempfile.mkdtemp(prefix="globule_glass_engine_")

# Independent storage manager
storage = SQLiteStorageManager(db_path=temp_dir / "tutorial.db")

# Fresh API instance
api = GlobuleAPI(storage=storage, orchestrator=orchestrator)
```

This ensures:
- No pollution of user's actual data
- Safe experimentation and testing
- Reproducible demonstrations
- Easy cleanup after completion

### Adding Custom Scenes

The scene-based architecture makes it easy to add new demonstrations:

```python
# In demo_scenes.py
async def demonstrate_new_feature(console: Console, api: GlobuleAPI):
    console.print(Panel.fit("[bold cyan]New Feature Demo[/bold cyan]"))
    
    with Progress(...) as progress:
        progress.add_task("Testing new feature...", total=None)
        result = await api.new_method()
    
    console.print(f"[bold green]Result: {result}[/bold green]")

# In interactive_mode.py  
async def execute_tutorial_flow(self) -> None:
    # ... existing scenes ...
    await demo_scenes.demonstrate_new_feature(self.console, self.api)
```

## Usage Scenarios

### For New Users
**Recommended**: Interactive Mode
- Shows the complete user journey
- Emphasizes practical benefits
- Builds confidence in the system
- Demonstrates AI capabilities

### For Stakeholders
**Recommended**: Interactive or Demo Mode
- Interactive for detailed presentations
- Demo for quick capability overview
- Both show professional polish
- Emphasize business value and AI intelligence

### For Developers
**Recommended**: Debug Mode + Interactive Mode
- Debug for technical understanding
- Interactive to see user perspective
- Both provide complete system visibility
- Debug shows performance characteristics

### For Testing
**All Modes**: Each run validates system functionality
- API method coverage
- Integration testing
- Performance benchmarking
- Error detection

## Best Practices

### Running Glass Engine
1. **Always run from clean state** - Glass Engine creates its own data
2. **Check Ollama availability** - System gracefully degrades to mocks if needed
3. **Use appropriate mode** - Match the mode to your audience
4. **Allow time for completion** - Interactive mode requires user input

### Customizing for Your Use Case
1. **Modify demo_scenes.py** - Add organization-specific examples
2. **Adjust timing** - Modify progress bar durations for effect
3. **Customize test data** - Use domain-relevant example thoughts
4. **Brand the presentation** - Update titles and descriptions

### Troubleshooting
- **Slow performance**: Check Ollama connectivity and model availability
- **Import errors**: Ensure all dependencies are installed
- **Database locks**: Glass Engine uses temporary databases to avoid conflicts
- **Rich display issues**: Ensure terminal supports rich console output

## Future Enhancements

The Glass Engine architecture supports easy extension:

- **Recording Mode**: Capture demonstrations for documentation
- **Benchmark Mode**: Systematic performance testing
- **Custom Scenarios**: Organization-specific use cases
- **Web Interface**: Browser-based Glass Engine for broader accessibility
- **Comparison Mode**: Before/after system demonstrations

The Glass Engine embodies Globule's core philosophy: transparency, education, and trust through visibility. It's not just a tutorial system - it's a fundamental part of how users understand and trust the application.