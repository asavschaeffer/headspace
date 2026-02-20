# Headspace UI Evolution

How the experience changes as Headspace matures — from "app I open" to "thing that's just there."

---

## Era 0: The Web Dashboard (Now)

**What it feels like:** A developer tool. You open a browser tab, click a button, wait, browse results.

```
┌─────────────────────────────────────────────────┐
│  headspace                    [Ingest] [Search]  │
├──────────────┬──────────────────────────────────┤
│ 📁 src/      │  main.rs                         │
│  ├─ main.rs  │  ─────────                       │
│  ├─ api.rs   │  Entry point for the server.     │
│  └─ ...      │  Cluster: "core infrastructure"  │
│ 📁 docs/     │                                  │
│  └─ ...      │  [Content preview...]            │
├──────────────┴──────────────────────────────────┤
│  ● ● ●    ●●   ●  ●●●   ● ●    2D cluster view │
│    ●  ● ●●   ●      ●●●    ●                    │
└─────────────────────────────────────────────────┘
```

**Who it's for:** You, right now. Proving the concept works.

**The good:** It exists. You can see clusters. Search works.
**The bad:** You have to *decide* to use it. You open a tab, run ingest, wait, then browse. It's a conscious act — like opening a filing cabinet instead of just *knowing where things are*.

**Ship it because:** The pipeline matters more than the interface right now. Web is the fastest way to iterate on the *backend* without fighting native UI frameworks.

---

## Era 1: The Quick Bar

**What it feels like:** Spotlight / Raycast / Alfred — but semantic.

The full dashboard still exists for deep exploration, but the **primary interaction** shrinks to a single bar:

```
   ╭──────────────────────────────────────────╮
   │  🔍  that rust thing for database stuff  │
   ╰──────────────────────────────────────────╯
          ┌──────────────────────────────┐
          │ 📄 storage.rs          94%   │
          │    JSON persistence layer    │
          │ 📄 useful-repos.md     87%   │
          │    jasonisnthappy reference  │
          │ 📄 Cargo.toml          71%   │
          │    hdbscan dependency        │
          └──────────────────────────────┘
                          ↵ Open  ⇥ Peek  ⌘N Neighborhood
```

**The key interaction:**
1. You press a **global hotkey** (e.g., `Ctrl+Space`)
2. A floating bar appears *over whatever you're doing*
3. You type a concept in plain language
4. Results appear ranked by semantic similarity
5. `Enter` opens the file. `Tab` shows a preview. `Ctrl+N` shows the semantic neighborhood.

**You never leave your workflow.** The bar appears, you find the thing, it disappears. 2 seconds total.

**What changes from Era 0:**
- Headspace runs as a **background daemon** (system tray) — no browser tab needed
- Ingest happens **automatically** on file changes (filesystem watcher)
- The web dashboard becomes the "power user mode" — you open it when you want to explore clusters, not when you want to find a file

**Hack version:** Tauri app with a global hotkey that opens a frameless window. Embeddings still from NVIDIA cloud. Watcher is a simple polling loop.

**Robust version:** Native OS hotkey registration. Local embeddings (Ollama). Proper `inotify`/`ReadDirectoryChangesW` filesystem watcher.

---

## Era 2: The Neighborhood View

**What it feels like:** You're looking at a star map of your project, and you can zoom into any constellation.

```
                    ┌─────────────────────────────────────┐
                    │         🟣 auth                      │
                    │       🟣   🟣                        │
    🔵 config       │     🟣  login.rs                     │
  🔵   🔵          │       🟣 oauth.rs                    │
    🔵              │                                      │
                    │              🟠 api                   │
                    │           🟠    🟠                    │
 🟤 docs            │         🟠  routes.rs                │
  🟤  🟤 🟤        │           🟠                          │
                    │                                      │
                    │  ⬜ storage.rs                        │
                    │     (noise — doesn't cluster)        │
                    └─────────────────────────────────────┘
                    
   Click a dot  →  peek file
   Drag to select  →  "export these as LLM context"
   Right-click cluster  →  "suggest folder for this group"
```

**What changes from Era 1:**
- The 2D view stops being a curiosity and becomes **actionable**
- You can *select* a region of the map and say "export this cluster as context for an LLM"
- Cluster labels are **auto-generated** (the system names them by summarizing their contents)
- The noise points (gray dots) become a **"rot detector"** — orphaned files that don't belong anywhere

**New interactions:**
- **Drag-select → Export:** Select a cluster, hit `Ctrl+E`, get a token-budgeted context dump
- **Right-click → Suggest Location:** "These 8 files are all about auth. Move to `src/auth/`?"
- **Hover → Peek:** See a file's content without leaving the map
- **Edge lines:** Optionally show the strongest semantic connections between files

---

## Era 3: The CLI & Pipes

**What it feels like:** Headspace is a Unix citizen. It composes with everything.

```bash
# Find files about a concept
$ hs find "error handling patterns"
src/api.rs           94%
src/storage.rs       87%
src/main.rs          72%

# Get LLM context for a question
$ hs context "how does ingestion work?" --budget 4000 | llm "summarize this"

# Show what changed semantically since last week
$ hs drift --since "1 week ago"
  [+] New cluster: "rate limiting" (3 files)
  [~] "auth" cluster grew: +2 files (oauth2 provider added)
  [-] "legacy-utils" cluster lost 4 files (deleted)

# Find orphaned files
$ hs orphans
  docs/old-notes.md       (no cluster, last modified 8 months ago)
  src/unused_helper.rs    (no cluster, 0 imports)

# Suggest a better directory layout
$ hs suggest-layout
  misc/tax-stuff.pdf  →  finances/2024/
  misc/sourdough.md   →  cooking/recipes/
  Apply? [y/n/preview]

# Pipe into fzf for fuzzy semantic search
$ hs find "database" | fzf --preview 'bat {}'
```

**What changes from Era 2:**
- Every feature from the GUI is also a **CLI command**
- The CLI becomes the **LLM's interface** — agents can call `hs find`, `hs context`, `hs suggest-layout`
- Headspace becomes a **building block**, not just an application

**Why this matters:** This is where Headspace stops being a *product* and becomes *infrastructure*. An MCP server, a VS Code extension, a Neovim plugin — they all talk to the same daemon through the CLI or a socket.

---

## Era 4: The Invisible Layer

**What it feels like:** You don't "use" Headspace. Your computer just... understands your files.

```
┌─ File Explorer ──────────────────────────────┐
│                                              │
│  📁 src/                                     │
│    📄 main.rs                                │
│    📄 api.rs          ← 94% related to ↑     │
│    📄 storage.rs                             │
│                                              │
│  ── Suggested ──────────────────────────      │
│  These files in other folders are related:    │
│    📄 docs/api-design.md        89%          │
│    📄 tests/api_test.rs         85%          │
│                                              │
├──────────────────────────────────────────────┤
│  Smart Folders (auto-generated)              │
│    🧠 Authentication (12 files)              │
│    🧠 Data Pipeline (8 files)                │
│    🧠 Configuration (5 files)                │
│    🧠 Orphaned (3 files) ⚠️                  │
└──────────────────────────────────────────────┘
```

**Key experiences:**
- **Smart Folders** appear alongside real folders — virtual groupings based on semantic clusters
- **"Related files"** show up automatically when you're looking at a file
- The **Save dialog** suggests where to save a new file based on its content
- **Spotlight/Search** is natively semantic — the OS search just *works* this way

**This is the Linus era.** Headspace isn't an app anymore. It's a filesystem feature.

---

## The Thread That Connects All Eras

Across every era, one thing stays constant: **the daemon and the index.**

```
Era 0:  Browser → HTTP → [Daemon + Index] → JSON file
Era 1:  Hotkey  → IPC  → [Daemon + Index] → Local DB
Era 2:  GUI     → IPC  → [Daemon + Index] → Local DB
Era 3:  CLI     → IPC  → [Daemon + Index] → Local DB
Era 4:  OS      → API  → [Daemon + Index] → Kernel/DB
```

The UI is disposable. **The daemon and the semantic index are the product.** Every era just wraps a different interface around the same core: "give me a concept, I'll give you the files."

This means:
- We can ship Era 0 now and it's useful
- We can build Era 1 without rewriting the backend
- We can add Era 3 alongside Era 1 (CLI + GUI coexist)
- Every UI we build is a **thin client** over the same semantic engine

---

## What We Build Next (and What We Don't)

| Priority | What | Why |
|----------|------|-----|
| **Now** | Harden the daemon (hash-based change detection, incremental re-embedding) | Foundation for everything — Era 1-4 all need this |
| **Soon** | `hs find` and `hs context` CLI | Fastest path to Era 3 usefulness, also enables LLM/MCP integration |
| **Soon** | Background filesystem watcher | Kills the "click Ingest" friction — critical for Era 1 |
| **Later** | Global hotkey quick bar (Tauri) | Era 1 — the moment it stops feeling like a "tool" |
| **Later** | Auto-generated cluster labels | Makes the neighborhood view and CLI output human-readable |
| **Much Later** | OS integration / smart folders | Era 4 — requires the index to be rock-solid first |

> **The rule: don't build a UI feature until the backend can support it reliably.** A beautiful quick bar that returns wrong results is worse than a ugly web page that returns right ones.
