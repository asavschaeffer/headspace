# Brain - Quick Start Guide

## 🚀 Launch the Galaxy

```bash
python start_brain.py
```

Your browser opens to `localhost:8888` - you're now in your living memory space.

## 🧠 How to Think

### Capture (Instant)
- Type anything in the input box
- Press `Ctrl+Enter` to save
- Watch the ripple as your thought enters orbit

### Search (Progressive)
- Type `/` followed by your search
- Results appear instantly: filenames → content → connections
- Click results to focus that memory in the galaxy

### Navigate (Spatial)
- Drag to pan through your memory space
- Scroll to zoom in/out
- Hover nodes to see previews
- Watch thoughts slowly drift together over time

## 📝 Command Line

```bash
# Quick save from terminal
brain "Just had an idea about caching strategies"

# Search your thoughts
brain --recent              # Last 24 hours
python brain "/search term"  # Search (use quotes on Windows)

# Launch web interface
brain                       # No arguments = start server
```

## 🌌 The Living System

Your thoughts aren't filed - they're **released into orbit**:

- **Similar thoughts drift together** over geological time
- **Recent memories glow brighter** and have more gravity
- **Connected ideas form constellations** you can see
- **The space breathes** with a 30-second rhythm

## 💭 Collision = Continuation

Same thought, different time? It **appends**, not replaces:

```
Original meeting notes...

==================================================
[Continued 2024-11-15 14:32:10]
==================================================

Follow-up thoughts...
```

Your meeting notes become a living document that grows.

## 🔬 Technical Details

### File Structure
```
~/brain/
  meeting-sarah-vector-db.txt     # Auto-named from content
  error-auth-bug.txt              # Semantic categorization
  code-calculate-distance.txt     # Code detection
  .metadata/                      # Connections & graph data
```

### Processing Pipeline
1. **Instant** (<10ms): File saved, visible in galaxy
2. **Fast** (<100ms): Searchable by content
3. **Background** (async): OCR, transcription, embeddings

### API Endpoints
- `POST /api/save` - Save thought
- `GET /api/graph` - Get galaxy data
- `GET /api/search?q=term` - Search
- `GET /api/recent` - Recent thoughts

## 🎯 Philosophy

This isn't organization - it's **emergence**.

After a week, you'll see:
- Clusters of related ideas
- Trails through your thinking
- Forgotten thoughts in deep orbit
- Active projects glowing bright

After a month:
- Solar systems of knowledge
- Asteroid belts of random thoughts
- Comets of recurring themes
- Black holes of obsessions

Your brain becomes a **living map of your mind**.

## 🛠️ Extend It

### Add processors for new file types
Edit `brain_core.py`:
```python
def process_markdown(self, filepath):
    # Extract headings, links, etc
    return searchable_text
```

### Tune the physics
Edit `brain_interface.html`:
```javascript
this.settings = {
    driftSpeed: 0.0001,      // Geological time
    breathingSpeed: 0.0002,  // 30 second cycle
    gravityStrength: 0.001   // Clustering force
}
```

### Connect external sources
The filesystem is the API - any tool can add thoughts:
```bash
echo "Thought from vim" > ~/brain/vim-note.txt
```

## 🌟 Tips

1. **Don't organize** - let gravity do it
2. **Type first, think later** - capture is instant
3. **Trust the drift** - similar thoughts find each other
4. **Watch the breathing** - it's intentionally calming
5. **Let old thoughts fade** - that's how memory works

Start typing. Watch your galaxy grow. 🌌