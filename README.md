# 🌌 Headspace - Cosmic Knowledge System

Transform your documents into navigable 3D constellations where chunks are stars, semantic relationships form nebulae, and knowledge becomes a cosmic memory palace.

---

## ⚡ Quick Start

### Option 1: Simple Setup (API Keys Only) ✨

```bash
# 1. Install lightweight dependencies (~50MB)
pip install -r requirements.txt

# 2. Copy and configure environment
cp env.example .env
# Edit .env and add your Gemini or OpenAI key

# 3. Run the application
python headspace/main.py
```

Then open: **http://localhost:8000**

### Option 2: Docker (All Features)

```bash
docker-compose up --build
```

### Option 3: Full Local Setup (With Local Models)

```bash
# Install core + local model support (~2-3GB)
pip install -r requirements.txt -r requirements-local.txt

# Install Ollama from https://ollama.ai
# Pull models: ollama pull gemma3:4b

python headspace/main.py
```

---

## ✨ Features

### 📄 Document View (Reading Mode)
- **Smart Chunking**: LLM-based intelligent chunking with fallback to structural
- **Attachment System**: Attach documents to specific chunks
- **Breadcrumb Navigation**: Track your path through document hierarchies
- **Subtle Interactions**: Hover shows parentheses `( text )` in margins - no text movement
- **Color-Coded**: Chunks colored by type and semantic embedding

### 🌌 Cosmos View (Exploration Mode)
- **3D Visualization**: Documents rendered as star systems in space
- **Semantic Clustering**: Similar chunks naturally cluster together
- **Nebulae**: Glowing particle clouds around related concepts
- **Interactive**: Hover to preview, click to zoom
- **Dramatic Effects**: Chunks glow and scale (1.5x hover, 1.8x selected)
- **Constellation Lines**: Sequential connections show document flow
- **Physics Simulation**: Optional gravity mode for organic movement

### 🤖 AI Features
- **Multiple Model Support**: Ollama, Gemini, OpenAI, Sentence-Transformers
- **Automatic Fallbacks**: Seamless switching between models if one fails
- **LLM Chunking**: Context-aware document chunking using language models
- **Semantic Embeddings**: High-dimensional vectors for similarity matching
- **Tag Generation**: Automatic tagging based on content analysis
- **Model Health Monitoring**: Real-time status tracking of all AI services

### 🔧 Developer Features
- **Modular Architecture**: Clean separation of concerns (API, Services, Models)
- **REST API**: Full CRUD operations on documents and chunks
- **SQLite Database**: Persistent storage with relationships
- **Real-time Updates**: WebSocket support
- **File Upload**: Secure file upload with validation
- **Comprehensive Testing**: Unit tests with pytest
- **Docker Support**: One-command deployment

---

## 🎮 Usage

### Adding Documents

1. Click **"+ Add Document"** in sidebar
2. Enter title and content (or upload file)
3. System automatically chunks and processes
4. View in document or cosmos mode

### Attaching Documents to Chunks

1. In document view, click the **pulsing badge** on any chunk
2. Click **"+ Attach Document"**
3. Enter document ID to attach
4. Attached documents appear in expandable panel

### Exploring in 3D

1. Switch to **Cosmos View**
2. **Hover** stars to see chunk previews (dramatic glow!)
3. **Click** to select and zoom camera
4. Info panel shows chunk details + attach button
5. Toggle nebulae, connections, gravity in controls

---

## 📁 Project Structure

```
.
├── headspace/                   # Main application package
│   ├── main.py                 # Application entry point
│   ├── api/                    # API layer
│   │   ├── routes.py          # FastAPI endpoints
│   │   └── middleware.py      # CORS, security headers
│   ├── services/              # Business logic
│   │   ├── database.py        # Database operations
│   │   └── document_processor.py  # Document processing
│   └── models/                # Data models
│       └── api_models.py      # Pydantic models
├── config_manager.py           # Configuration management
├── embeddings_engine.py        # Embedding generation
├── llm_chunker.py             # LLM-based chunking
├── tag_engine.py              # Document tagging
├── model_monitor.py           # Model health monitoring
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── env.example                # Environment variables template
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Container build
├── static/
│   └── index.html            # Web frontend (Three.js visualization)
├── documents/                  # Document storage
├── templates/                  # HTML templates for standalone export
├── scripts/                    # Launcher scripts
│   ├── start_headspace.bat
│   └── run_docker.bat
├── docs/                       # Detailed documentation
│   ├── HEADSPACE.md
│   ├── DOCKER.md
│   └── ...
├── legacy/                     # Old loom-based system (archived)
└── archive/                    # Experiments & prototypes
```

---

## 🔧 Technical Stack

**Backend**:
- Python 3.11+
- FastAPI (web framework)
- SQLite (database)
- NumPy (embeddings & positioning)

**Frontend**:
- Three.js r128 (3D rendering)
- OrbitControls (camera navigation)
- Custom GLSL shaders (nebula effects)
- Vanilla JavaScript (no framework)

**Deployment**:
- Docker & Docker Compose
- Uvicorn ASGI server
- Volume mounts for persistence

---

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve web interface |
| `POST` | `/api/documents` | Create document |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document with chunks |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `POST` | `/api/chunks/{id}/attach` | Attach document to chunk |
| `GET` | `/api/chunks/{id}/attachments` | Get chunk attachments |
| `DELETE` | `/api/chunks/{id}/attach/{doc_id}` | Remove attachment |
| `GET` | `/api/visualization` | Get all data for 3D viz |
| `POST` | `/api/upload` | Upload file |
| `GET` | `/api/health` | Basic health check |
| `GET` | `/api/health/models` | Model status overview |
| `GET` | `/api/health/detailed` | Comprehensive service check |

---

## 🎨 Visual Language

### Document View
- **Chunk borders**: Color-coded by type (red=header, blue=paragraph, purple=code)
- **Attachment badges**: Circular, left edge, pulses when has attachments
- **Hover**: Subtle parentheses `( ... )` fade in at margins
- **No text movement**: Reading experience never interrupted

### Cosmos View
- **Star size**: Larger = more text
- **Star color**: From semantic embedding (similar colors = related concepts)
- **Star position**: UMAP reduction (close together = semantically similar)
- **Nebulae**: Particle clouds around semantic clusters
- **Constellation lines**: Show document structure
- **Hover glow**: Scale 1.5x, emissive intensity 1.2x
- **Selected glow**: Scale 1.8x, emissive intensity 1.5x

---

## 🔧 Configuration

### API Keys (At Least One Required)

1. **Copy `env.example` to `.env`**:
   ```bash
   cp env.example .env
   ```

2. **Add your API keys**:
   - `GEMINI_API_KEY` - Google Gemini (recommended, free tier available)
   - `OPENAI_API_KEY` - OpenAI GPT models
   - `OLLAMA_URL` - Local Ollama (optional, for offline use)

### Dependency Options

| File | Size | Use Case |
|------|------|----------|
| `requirements.txt` | ~50MB | Basic setup with API keys (Gemini/OpenAI) |
| `requirements-local.txt` | ~2-3GB | Adds local embedding models (Sentence-Transformers) |
| `requirements-dev.txt` | ~10MB | Adds testing tools (pytest, coverage) |

### Model Configuration

Edit `loom_config.json` to:
- Set preferred models and providers
- Configure fallback chains
- Enable/disable specific services

---

## 🚀 Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python headspace/main.py

# Server runs on http://localhost:8000
```

### Frontend Development

Edit `static/index.html` - changes are live (volume mounted in Docker)

### Backend Development

Edit files in `headspace/` directory - restart container to apply changes:
```bash
docker-compose restart
```

---

## 🧪 Testing

### Running Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=headspace
```

### Manual Testing
1. Start server: `docker-compose up`
2. Open: http://localhost:8000
3. Add a document
4. Switch between views
5. Test attachments
6. Explore cosmos

### API Testing
```bash
# Health check
curl http://localhost:8000/api/health

# Model status
curl http://localhost:8000/api/health/models

# List documents
curl http://localhost:8000/api/documents

# Get specific document
curl http://localhost:8000/api/documents/{doc_id}

# Attach document to chunk
curl -X POST http://localhost:8000/api/chunks/{chunk_id}/attach \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc_id"}'
```

---

## 🎯 Philosophy

**Memory Palace Principle**: Humans remember spatial arrangements better than linear text. Headspace leverages spatial memory by placing ideas in 3D space.

**Semantic Gravity**: Related concepts naturally attract, different concepts repel. Clusters emerge organically from meaning.

**Progressive Detail**: Far view shows structure, near view shows content. Navigate smoothly between macro understanding and micro detail.

**Context-Aware UX**: Reading mode is subtle and respectful. Exploration mode is dramatic and engaging.

---

## 🗺️ Roadmap

### ✅ Phase 1: Foundation (Complete)
- Document chunking & storage
- 3D visualization with Three.js
- Attachment system
- Breadcrumb navigation

### 🔄 Phase 2: Beautiful Nebulae (In Progress)
- 2000-particle nebula systems
- Custom GLSL shaders
- Nebula core glow
- Better cluster detection

### 📋 Phase 3: Hierarchical Exploration
- Click-to-expand node hierarchy
- Document → Chunks → Attachments drill-down
- Smooth camera zoom animations
- Visual hierarchy by level

### 🔮 Future
- Search across all documents
- Real embeddings (OpenAI/Cohere)
- Multi-user support
- VR support for immersive exploration
- Export visualizations

---

## 📜 License

MIT - Free to use and modify

---

## 🆘 Troubleshooting

**Server won't start?**
- Check Python 3.11+ installed
- Install dependencies: `pip install -r requirements_headspace.txt`

**Docker issues?**
- Ensure Docker Desktop is running
- Check port 8000 is free: `netstat -ano | findstr :8000`

**Database errors?**
- Delete `headspace.db` for fresh start
- Container will create new database automatically

**More help**: See [docs/](docs/) for detailed guides

---

**Built with 💫 - Explore your knowledge as a cosmic memory palace**
