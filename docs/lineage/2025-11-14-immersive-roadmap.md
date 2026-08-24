\# Headspace Immersive Roadmap



\## Purpose

\- Establish the next wave of features that turn Headspace into a multiplayer, living knowledge cosmos.

\- Align on the graph-powered backend expansion and the immersive visual layer (skybox, constellations, hover FX).

\- Provide a phased implementation plan with dependencies, risks, and open questions.



\## Current Foundations

\- \*\*Semantics \& Layout\*\*: chunks enriched with embeddings, cosine-positioned coordinates, optional UMAP/HDBSCAN pipeline.

\- \*\*Renderer\*\*: Three.js cosmos with geometry LOD, enrichment streaming, and home-planet onboarding.

\- \*\*Storage\*\*: Supabase or SQLite for documents/chunks; enrichment events via WebSocket.

\- \*\*Assets Ready\*\*: procedural geometry worker, color theming, and seed documents that define the “tutorial” hub.



\## Graph Layer Expansion

\### Goals

\- Promote ideas, users, clusters, and relationships to first-class graph entities.

\- Enable multi-hop queries (e.g., “which signatures bridge cluster A and B?”).

\- Drive constellation rendering and gameplay (quests, lore anchors) directly from graph data.



\### Data Model (Initial Types)

\- \*\*Nodes\*\*: `User`, `Signature`, `Document`, `Chunk`, `Cluster`, `Tag`, optional `PlanetGeometry`.

\- \*\*Edges\*\*:

&nbsp; - `CREATED\_BY` (User → Document/Chunk)

&nbsp; - `SIGNED\_BY` (Signature → Document)

&nbsp; - `BELONGS\_TO` (Chunk → Document)

&nbsp; - `IN\_CLUSTER` (Chunk → Cluster)

&nbsp; - `SIMILAR\_TO` (weighted Chunk ↔ Chunk)

&nbsp; - `LINKS\_TO` (Document ↔ Document, e.g., attachments)

&nbsp; - `INSPIRED\_BY` / `MENTIONS` (optional narrative edges)

&nbsp; - `NEAREST\_NEIGHBOR` (Chunk ↔ Chunk, limited fan-out for LOD)



\### Infrastructure Options

1\. \*\*Neo4j Aura\*\* (preferred): managed, Cypher support, strong tooling.

2\. \*\*Supabase + Apache AGE\*\*: remain within Postgres if we want fewer services (trade-off: less mature tooling).

3\. \*\*RedisGraph / Memgraph\*\*: high throughput for real-time leaderboards; best if we prioritize streaming queries.



\### Sync \& Operations

\- \*\*Ingestion\*\*: extend enrichment pipeline (`DocumentProcessor`) to emit graph upserts (async task or event queue).

\- \*\*Backfill\*\*: one-off script that walks current DB and populates graph (batched bulk writes).

\- \*\*Change Data Capture\*\*: subscribe to Supabase real-time or use a job queue to update graph when documents mutate.

\- \*\*API Wrapper\*\*: add `graph\_repository.py` for typed helpers (e.g., `get\_constellation(document\_id)`).



\### Surface Area

\- Expose new endpoints (or expansions) to request cluster summaries, player neighborhoods, or recommended paths.

\- Feed graph-derived data to renderer: highlight high-degree nodes, badges for signature hubs, quest prompts.



\## Visual Immersion Upgrades

\### Skybox \& Atmosphere

\- Load a lightweight HDRI for deep-space ambience; fallback to procedural shader if texture unavailable.

\- Ensure WebGL1 compatibility and minimal VRAM (compressed KTX2 + JPG fallback).

\- Integrate with existing fog/tone mapping so planets feel embedded in a true galaxy.



\### Constellation \& Bond Effects

\- Replace simple `THREE.Line` connections with `THREE.Line2` (fat-line) or instanced ribbons.

\- Color palette: pale opalescent lines (rgba ~200/220/255/0.25); animate alpha when hovered/selected.

\- Use graph-derived edges to limit clutter (nearest neighbors, quest paths, cluster centroids).

\- Add optional pulsing particles that travel along edges to signal thought streams.



\### Hover \& Selection Feedback

\- Introduce additive halo sprites (`THREE.Sprite`) that fade in/out on hover.

\- Add optional outline/inner-glow post-processing (Sobel or OutlinePass) with tight thresholds to debounce noise.

\- Enhance tooltips: floating mini-panel near cursor for quick lore; keep side panel for deep info.



\### Additional Visual Touches

\- Camera bloom/tonemapping tweaks via `EffectComposer` (kept optional for low-end devices).

\- Particle nebulae per cluster derived from graph communities (controlled by settings).

\- Ambient audio hooks (future) triggered by proximity to specific clusters or signatures.



\## Implementation Roadmap

1\. \*\*Phase 1 – Visual Foundations\*\*

&nbsp;  - Introduce skybox loader (`skybox.js`) + integrate into `initCosmos()`.

&nbsp;  - Add hover halo sprites and migrate existing hover logic to new helper.

&nbsp;  - Refactor connection rendering to support pale line styling (still using current data).

2\. \*\*Phase 2 – Graph Backbone\*\*

&nbsp;  - Choose graph engine and provision environment.

&nbsp;  - Implement enrichment emitters + backfill script.

&nbsp;  - Expose graph queries for renderer (e.g., `get\_constellation\_paths`).

3\. \*\*Phase 3 – Constel Gameplay\*\*

&nbsp;  - Switch line rendering to graph-driven edges.

&nbsp;  - Add quest/prompt overlays using graph analytics (betweenness, shortest paths).

&nbsp;  - Introduce user identity/signature mapping to nodes.

4\. \*\*Phase 4 – Multiplayer Polish\*\*

&nbsp;  - Realtime graph updates (broadcast new planets, orbit shifts).

&nbsp;  - Presence indicators, shared cursor trails, and optional chat/lore modules.

&nbsp;  - Performance pass: culling, dynamic LOD per cluster, asset compression.



\## Dependencies \& Risks

\- Graph service adds operational overhead—need infra plan (backups, monitoring).

\- Visual effects must preserve current performance; require fallback path for low-end GPUs.

\- Real-time features will need auth/session groundwork to prevent abuse.

\- HDR/skybox assets must be licensed appropriately for public web distribution.



\## Open Questions

\- Which graph engine fits best with current hosting constraints?

\- Do we gate graph writes behind queues to avoid blocking document creation?

\- Where should user identity live (Supabase auth, custom JWT, wallet-based signatures)?

\- How aggressive do we want default visual effects versus an accessibility/“performance mode”?



\## Next Steps

1\. Confirm graph engine choice and provisioning timeline.

2\. Approve visual priorities (skybox vs. constellations first) and asset direction.

3\. Schedule implementation sprints aligned with the roadmap phases.

4\. Start drafting API contracts for graph queries the renderer will consume.





