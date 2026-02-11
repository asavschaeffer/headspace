# headspace

**Make computer information organization and retrieval easier — for humans and for LLMs.**

Headspace is a Rust-powered tool that scans a directory of files, generates semantic embeddings for each one, clusters them by meaning, and serves a web UI where you can browse, search, and visualize everything in 2D.

---

## Why

Files on your computer are organized by *where* you put them — folders, names, dates. But that rarely reflects *what's in them* or *how they relate* to each other.

Headspace flips this: it reads your files, understands their content through embeddings, and groups them by meaning. The result is a searchable, visual map of your information — not just a file tree.

This is useful for:
- Finding documents you forgot you had
- Seeing how your notes, code, and configs relate to each other
- Giving an LLM structured context about a codebase or project
- Reorganizing directories based on actual content, not habit

---

## How It Works

```
Directory on Disk
       │
       ▼
┌──────────────┐
│  Ingestion   │  Recursively crawl files, extract text
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embeddings  │  NVIDIA NIM API (nv-embedqa-e5-v5, 1024-dim)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Clustering  │  HDBSCAN — density-based, no predetermined k
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Projection  │  PCA → 2D coordinates for visualization
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Storage     │  JSON file (.headspace/store.json)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Axum Server │  REST API + static frontend
└──────┬───────┘
       │
       ▼
   Web UI at localhost:3000
```

### Pipeline in Detail

1. **Ingestion** — Walks the target directory recursively. Reads text-based files (`.md`, `.rs`, `.py`, `.js`, `.json`, `.toml`, `.yaml`, `.txt`, `.html`, `.css`, and 30+ more). Skips binaries, hidden directories, `node_modules`, `target`, `__pycache__`, etc. Max file size: 1 MB.

2. **Embeddings** — Sends file content (truncated to ~2000 chars) to the [NVIDIA NIM](https://build.nvidia.com/) embedding API in batches of 50. Uses `nv-embedqa-e5-v5` (1024 dimensions). Differentiates between "passage" embeddings (for documents) and "query" embeddings (for search) for better relevance. Falls back to zero vectors if no API key is set.

3. **Clustering** — Runs [HDBSCAN](https://docs.rs/hdbscan) on the embedding matrix. HDBSCAN is density-based: it doesn't require you to pick a number of clusters, it handles noise naturally, and it finds clusters of varying density. Documents that don't fit any cluster are labeled as "noise" (`cluster_id = -1`).

4. **2D Projection** — Projects the high-dimensional embeddings (1024D) down to 2D using PCA (power iteration for the top 2 principal components). Coordinates are normalized to `[0, 1]` for rendering on a canvas.

5. **Storage** — Everything is persisted to a single JSON file at `.headspace/store.json`. Each document stores: ID, paths, name, extension, content preview, embedding vector, cluster ID, 2D coordinates, modification time, and ingestion timestamp.

6. **Server** — An [Axum](https://docs.rs/axum) HTTP server serves the REST API and the static frontend. All state is held in memory behind `Arc<RwLock<>>` for safe concurrent access. Ingestion runs as a background `tokio::spawn` task so the UI stays responsive.

---

## Setup

### Prerequisites

- **Rust** (stable, 2024 edition) — [rustup.rs](https://rustup.rs/)
- **NVIDIA NIM API key** (free tier available) — [build.nvidia.com](https://build.nvidia.com/)

### Installation

```bash
git clone <this-repo>
cd headspace-5
```

Create a `.env` file in the project root:

```
NVIDIA-API-KEY = nvapi-your-key-here
```

Build and run:

```bash
cargo run
```

Open [http://localhost:3000](http://localhost:3000).

### Configuration

All config is via environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA-API-KEY` | *(none)* | NVIDIA NIM API key for embeddings |
| `HEADSPACE_PORT` | `3000` | Port for the web server |
| `HEADSPACE_DATA_DIR` | `.headspace` | Directory for persistent storage |

> Without an API key, headspace will still ingest and display files, but search and clustering won't be meaningful (all embeddings will be zero vectors).

---

## Usage

### 1. Ingest a Directory

Click the **Ingest** button in the header, enter a directory path, and hit **Start Ingestion**.

The server will:
- Crawl the directory recursively
- Extract text from supported files
- Generate embeddings via NVIDIA NIM
- Run HDBSCAN clustering
- Compute 2D projection coordinates
- Save everything to `.headspace/store.json`

The UI polls for status every 2 seconds and refreshes automatically when ingestion completes.

### 2. Browse Files

The left panel shows an interactive file tree. Click any file to see its content and metadata in the right detail panel.

### 3. Search

Switch to the **Search** tab and type a natural language query. Headspace generates an embedding for your query and ranks all documents by cosine similarity. Results show the relevance score as a percentage.

This is *semantic* search — it matches by meaning, not keywords. Searching "error handling" will find files about exceptions, panics, fallbacks, etc., even if they never use those exact words.

### 4. Visualize Clusters

The **Clusters** tab shows a 2D scatter plot of all documents. Each dot is a file, colored by its HDBSCAN cluster. Hover for tooltips, click to open the detail view.

Documents that are semantically similar appear close together. Noise points (unclustered) are shown in gray.

---

## REST API

The server exposes these endpoints:

| Method | Route | Body / Params | Response |
|--------|-------|---------------|----------|
| `GET` | `/api/status` | — | `{ document_count, has_embeddings, is_ingesting, root_path }` |
| `POST` | `/api/ingest` | `{ "path": "C:\\..." }` | `{ message, document_count }` |
| `GET` | `/api/documents` | — | `[{ id, name, rel_path, extension, content_length, cluster_id }]` |
| `GET` | `/api/document/:id` | — | Full document object (without embedding vector) |
| `GET` | `/api/search` | `?q=...&limit=20` | `[{ id, name, rel_path, score, content_preview, cluster_id }]` |
| `GET` | `/api/clusters` | — | `[{ id, name, rel_path, extension, cluster_id, x, y }]` |

---

## Project Structure

```
headspace-5/
├── Cargo.toml              # Dependencies and project metadata
├── .env                    # API keys (gitignored)
├── .env.example            # Template for .env
├── .gitignore
│
├── src/
│   ├── main.rs             # Entry point, server startup
│   ├── config.rs           # Environment/config loading
│   ├── ingestion.rs        # Directory crawler + text extraction
│   ├── embeddings.rs       # NVIDIA NIM embedding API client
│   ├── storage.rs          # Document model + JSON persistence
│   ├── search.rs           # Cosine similarity search
│   ├── cluster.rs          # HDBSCAN clustering + PCA projection
│   └── api.rs              # Axum REST API routes + handlers
│
├── frontend/
│   ├── index.html          # Single-page app shell
│   ├── style.css           # Dark-mode design system
│   └── app.js              # UI logic, canvas visualization, API calls
│
├── docs/
│   ├── user-story.md       # Original vision and pipeline design
│   ├── useful-repos.md     # Reference repositories and crates
│   └── rust-guidelines.md  # Microsoft Pragmatic Rust Guidelines
│
└── .headspace/             # Runtime data (gitignored)
    └── store.json          # Persisted document store
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | **Rust** (2024 edition) | Performance, safety, single binary |
| HTTP Server | **Axum** + Tokio | Async, ergonomic, production-grade |
| Embeddings | **NVIDIA NIM** (`nv-embedqa-e5-v5`) | High-quality 1024-dim embeddings, free tier |
| Clustering | **HDBSCAN** | Density-based, no fixed k, handles noise |
| 2D Projection | **PCA** (power iteration) | Lightweight, no extra dependencies |
| Storage | **JSON file** | Zero-dependency persistence, human-readable |
| Allocator | **mimalloc** | Faster allocation for data-heavy workloads |
| Error Handling | **eyre** + color-eyre | Rich, contextual error reports |
| Logging | **tracing** | Structured, async-aware logging |
| Frontend | **Vanilla HTML/CSS/JS** | No build step, no npm, instant dev cycle |

### Key Dependencies

```toml
axum = "0.8"           # HTTP framework
tokio = "1"            # Async runtime
hdbscan = "0.12"       # Clustering
reqwest = "0.12"       # HTTP client for NVIDIA API
walkdir = "2"          # Recursive directory traversal
serde = "1"            # Serialization
mimalloc = "0.1"       # Global allocator
eyre = "0.6"           # Error handling
tracing = "0.1"        # Structured logging
```

---

## Supported File Types

Headspace reads any UTF-8 text file with these extensions:

> `.txt` `.md` `.rs` `.py` `.js` `.ts` `.jsx` `.tsx` `.json` `.toml` `.yaml` `.yml` `.html` `.css` `.scss` `.c` `.cpp` `.h` `.hpp` `.go` `.java` `.sh` `.bat` `.ps1` `.xml` `.csv` `.log` `.cfg` `.ini` `.env` `.sql` `.rb` `.php` `.swift` `.kt` `.r` `.lua` `.pl` `.ex` `.exs` `.zig` `.nim` `.v` `.d` `Makefile` `Dockerfile`

Binary files, files over 1 MB, and hidden directories are automatically skipped.

---

## Design Decisions

**Why JSON storage instead of a database?**
For an MVP, a single JSON file is the simplest thing that works. It avoids database dependencies, is human-readable, and handles ~10K documents with sub-second load times. The storage layer is abstracted so it can be swapped for something like [jasonisnthappy](https://github.com/sohzm/jasonisnthappy) later.

**Why PCA instead of UMAP?**
UMAP produces better 2D projections but requires native C dependencies. PCA is trivial to implement in pure Rust (50 iterations of power iteration) and gives a usable scatter plot for MVP. UMAP can be added via the [fast-umap](https://crates.io/crates/fast-umap) crate later.

**Why HDBSCAN?**
Unlike K-Means, HDBSCAN doesn't require you to guess the number of clusters. It handles noise naturally — not every document needs to belong to a cluster. It finds clusters of varying density, which matches real-world file collections where some topics have many files and others have few.

**Why a server instead of a CLI?**
The 2D visualization and interactive search need a browser. A server lets the frontend call the backend via REST without any build tooling. The same API can later serve LLM agents or IDE extensions.

---

## Inspiration

- [**BabyAGI 3**](https://github.com/yoheinakajima/babyagi3) — Modular agent architecture with listeners, memory layers, and extensible tools
- [**jasonisnthappy**](https://github.com/sohzm/jasonisnthappy) — Embeddable Rust document database with ACID transactions and full-text search
- [**hdbscan**](https://docs.rs/hdbscan) — Density-based clustering that handles noise and varying cluster densities

---

## Roadmap

- [ ] Multimodal ingestion (images via Moondream API, PDFs)
- [ ] Chunked embeddings (paragraph-level, not just file-level)
- [ ] UMAP for better 2D projections
- [ ] LLM-powered directory restructuring suggestions
- [ ] Re-ingestion with change detection (only re-embed modified files)
- [ ] Tag generation and knowledge graph
- [ ] Link/URL ingestion
- [ ] Export for LLM context windows

---

## License

MIT
