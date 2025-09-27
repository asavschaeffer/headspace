# Brain - Living Memory Galaxy

A filesystem-based thought capture system with a breathing, orbital visualization. Type anything, find it later through any mental pathway.

## Quick Start

```bash
python start_brain.py
```

Opens browser to `http://localhost:8888` with the galaxy interface.

## Features

### Core Principles
- **Instant Capture**: <10ms save latency - thought speed input
- **Collision as Continuation**: Same filename = same thought thread (echo-append)
- **Living Organization**: Files drift together based on similarity over geological time
- **Progressive Search**: Instant filename → fast content → semantic similarity

### The Breathing Space
- Files orbit slowly like cosmic dust
- Similar thoughts naturally cluster
- Recent activity creates ripples
- The entire space breathes with a 30-second rhythm

## Usage

### Input Modes
- **Save**: Type anything, press `Ctrl+Enter`
- **Search**: Start with `/` to search
- **Navigate**: Click and drag to pan, scroll to zoom

### File Organization
```
~/brain/
  meeting-sarah-vector-db.txt     # Auto-named from content
  error-auth-middleware.txt       # Semantic detection
  question-how-to-implement.txt   # Question detection
  code-calculate-distance.txt     # Code detection
  [.metadata/]                    # Connections and metadata
```

### Collision Handling
When you save with the same filename, it appends with a timestamp separator:
```
Original thought about vectors...

==================================================
[Continued 2024-11-15 14:32:10]
==================================================

New thought about vectors...
```

## Technical Architecture

### Components
1. **brain_core.py**: Filesystem operations, collision handling, metadata
2. **brain_server.py**: HTTP server with REST API
3. **brain_interface.html**: Living galaxy visualization
4. **start_brain.py**: Launcher with auto-browser open

### Processing Pipeline
```
Input → Detect Type → Generate Filename → Check Collision → Save/Append
                                                ↓
                                    Extract Metadata → Update Graph
```

### Search Layers
1. **Instant** (<10ms): Filename matching
2. **Fast** (<100ms): Content grep/ripgrep
3. **Smart** (background): Semantic similarity

### Orbital Mechanics
- **Mass**: File size determines gravitational pull
- **Heat**: Recent access makes nodes glow
- **Age**: Older files fade and move slower
- **Connections**: Similar content creates attraction

## API Endpoints

- `GET /` - Main interface
- `POST /api/save` - Save new thought
- `GET /api/graph` - Get nodes and edges
- `GET /api/search?q=query` - Search content
- `GET /api/file/{name}` - Get file content
- `GET /api/recent` - Recently modified files

## Extending

### Add Processors
Edit `brain_core.py` to add new file type processors:
```python
def process_newtype(self, filepath):
    # Extract searchable text
    return extracted_text
```

### Customize Physics
Edit orbital mechanics in `brain_interface.html`:
```javascript
this.settings = {
    driftSpeed: 0.0001,      // Geological time drift
    gravityStrength: 0.001,   // Center attraction
    connectionAttraction: 0.01 // Edge pull
}
```

## Philosophy

This isn't a filing system - it's a living memory space. Files aren't stored, they're released into orbit. Similar thoughts converge naturally. The visualization shows your mind's organizational patterns emerging without conscious effort.

After a month, you'll see:
- Solar systems of related thoughts
- Asteroid belts of scattered ideas
- Comets of recurring themes
- Dark matter of forgotten memories

## Requirements

- Python 3.6+
- Modern browser with Canvas support
- Optional: ripgrep for faster search
- Optional: tesseract for OCR
- Optional: whisper for audio transcription

## License

MIT - Your thoughts belong to you.