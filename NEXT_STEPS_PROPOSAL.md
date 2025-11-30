# Proposed Next Steps: Visuals, Performance, & Search Infrastructure

## 1. Visuals & Immersion
**Goal:** Enhance the "cosmic" feel while maintaining clarity.
- [ ] **Post-Processing Pipeline:** Implement `EffectComposer` in Three.js to add:
    - **Bloom/Glow:** Make active chunks and the home planet glow softly.
    - **Film Grain:** Slight grain to reduce color banding in the dark background.
- [ ] **Dynamic Starfield:** Upgrade the static starfield to use shaders that twinkle or parallax move slightly based on camera depth.
- [ ] **Connection Lines:** Render thin lines between related chunks (using the `connections` data) to visualize the knowledge graph topology.
- [ ] **Smooth Transitions:** Add camera ease-in/out animations when switching between "Home" and "Search Results" views.

## 2. Performance & Loading
**Goal:** Instant interactivity for larger datasets (>1000 chunks).
- [ ] **Instanced Rendering:** Move from individual `THREE.Mesh` objects for chunks to `THREE.InstancedMesh`. This is the single biggest performance unlock for rendering thousands of nodes.
- [ ] **Texture Compression:** Convert the Home Planet GLB to use Draco compression to reduce download size.
- [ ] **Spatial Indexing (Octree):** Implement a spatial index on the client-side to only raycast/check collisions against nearby chunks, speeding up mouse interaction.
- [ ] **Web Worker Layout:** Move the heavy force-directed graph layout or UMAP position calculations to a Web Worker so the UI thread never freezes.

## 3. Keyword Infrastructure
**Goal:** Robust, typo-tolerant text matching.
- [ ] **Persistent Indexing:** Serialize the `KeywordSearchEngine` index to disk (e.g., pickle or JSON) so it doesn't need to be rebuilt from the DB on every server restart.
- [ ] **Fuzzy Matching:** Integrate `thefuzz` or Levenshtein distance to handle typos in user queries.
- [ ] **Stopword Tuning:** Refine the exclusion list (stopwords) to better match the specific domain vocabulary of your "headspace."

## 4. Vector Search
**Goal:** Higher relevance and scalability.
- [ ] **HNSW Index:** Replace the flat Numpy vector scan with an HNSW (Hierarchical Navigable Small World) index (via `faiss` or `usearch`) for sub-millisecond retrieval at scale.
- [ ] **Embedding Cache:** Cache frequently queried embeddings (e.g., "finance", "coding") to skip the API call to the embedding model.
- [ ] **Hybrid Weight Tuning:** Create a small evaluation script to auto-tune the `vector_weight` vs. `keyword_weight` based on a set of test queries (Golden Set).

## 5. Search Infrastructure (General)
**Goal:** A unified, responsive search API.
- [ ] **Search History & Suggestions:** Store recent search queries in `localStorage` or the DB to offer autocomplete suggestions.
- [ ] **Filter Facets:** Add UI toggles to filter search results by `doc_type` (code vs. text) or `date_created`.
- [ ] **Async Index Updates:** Decouple document ingestion from indexing. When a file is uploaded, return "Processing..." immediately and update the search index in the background via a queue.

