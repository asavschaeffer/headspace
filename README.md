# Thoughtspace - Cosmic Document Explorer (v0.1)

## The Ultra-Minimal Version

This is the simplest possible implementation of your cosmic document visualization concept. It focuses on the core magic: **text becomes a star in semantic space**.

## What it does

- **Drop text** → Creates a glowing sphere node
- **Embedding magic** → Text is converted to vectors using MiniLM (runs in browser!)
- **Semantic positioning** → Similar concepts appear near each other in 3D space
- **Living space** → Nodes gently drift, making the space feel alive
- **Beautiful starfield** → Creates an immersive cosmic atmosphere

## How to run

1. Start the server:
```bash
python server.py
```

2. Open in your browser:
```
http://localhost:8080
```

## How to use

- **Type or paste text** in the input box and press Enter
- **Drag and drop** text files onto the drop zone
- **Paste text** anywhere on the page (Ctrl+V)
- **Navigate** with mouse: drag to rotate, scroll to zoom

## Technical details

- **Pure client-side**: Everything runs in your browser, no backend needed
- **Embeddings**: Uses Xenova/transformers.js with MiniLM model
- **3D engine**: Three.js for WebGL rendering
- **Zero dependencies**: Just one HTML file + CDN scripts

## What's next?

Once we validate this core experience works well, we can add:
- Image support (drag images to create visual nodes)
- Connections between nodes (citations, links)
- Better dimensionality reduction (UMAP instead of simple projection)
- Persistence (save/load your thoughtspace)
- Multi-user collaboration
- Neural network dynamics

But for now, enjoy the simple magic of watching your thoughts float in space!