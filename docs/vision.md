# Headspace: Endgame Vision

## The Idea

**Your computer already knows what your files *are*. It just can't tell you.**

Headspace makes your filesystem *searchable by meaning* — so the answer to "where did I put that?" is always one sentence away.

---

## The Endgame UX

The endgame is not a "tool you open." It's an **ambient layer** — always running, always indexed, always reachable. You never "organize" files again. You just *ask*.

Think of it like Spotlight on macOS, but instead of matching keywords, it understands concepts. And instead of just finding files, it *sees the relationships between them*.

### What It Feels Like

- You press a hotkey. A bar appears.
- You type: `"that rust crate I was looking at for database stuff"`
- Three files appear. The one you forgot existed is first.
- You hit Enter. It opens.

That's it. That's the product.

Everything else — clustering, visualization, reorganization suggestions, LLM context export — is built *on top* of that one interaction getting right.

---

## User Stories

### 🔍 1. "Where's That Thing?"
> *As a developer, I want to describe a file by what it's about — not by its name or path — and have it found instantly.*

**Today:** You open Explorer, click through 6 folders, realize it's not there, try `grep`, try `Everything`, give up, rewrite it from memory.

**Endgame:** You press `Ctrl+Space`, type `"the config where I set the database timeout"`, and it's there. 200ms.

---

### 🧠 2. "Show Me the Neighborhood"
> *As a researcher, I want to see every file conceptually related to the one I'm looking at, even across unrelated folders.*

**Today:** You have notes in `~/Documents`, code in `~/Projects`, and a PDF in `~/Downloads`. They're all about the same topic. You'd never know.

**Endgame:** You right-click a file → "Show Related." A constellation appears — your notes, the code, the PDF, a config file you forgot about — all clustered by meaning, not by folder. You see gaps in your knowledge. You see duplicates.

---

### 🏗️ 3. "My Folders Are a Lie"
> *As a power user, I want the system to show me how my files SHOULD be organized based on what they actually contain.*

**Today:** You have 14 folders named variations of "Projects", "Old", "Misc", "TODO", and "asdfasdf." Your folder structure reflects your mood on the day you saved the file, not the file's actual content.

**Endgame:** You run `headspace suggest-layout`. It shows you:
```
Your folder "misc/" contains 3 distinct clusters:
  → 12 files about "Rust async patterns" (suggest: ~/knowledge/rust/async/)
  → 8 files about "tax receipts 2024"    (suggest: ~/finances/2024/)
  → 3 files about "sourdough recipes"    (suggest: ~/cooking/)
  
Apply this plan? [preview / apply / skip]
```
It's not prescriptive. It shows you *what your files think they are* and lets you decide.

---

### 🤖 4. "Brief the LLM"
> *As an AI-assisted developer, I want to give an LLM the most relevant context about my project — automatically, without manually curating files.*

**Today:** You paste 15 files into ChatGPT manually. You miss the important one. The LLM hallucinates because it didn't have enough context.

**Endgame:** You run `headspace context "how does the auth system work?"` and it outputs a semantically-coherent, token-budgeted slice of your codebase — the files that *matter* for that question, in the right order, with cluster labels as section headers. You pipe it straight into your LLM.

```bash
headspace context "how does auth work?" --budget 8000 | llm "find the bug in the login flow"
```

---

### 📡 5. "What Changed — Conceptually?"
> *As a team lead, I want to know not just WHICH files changed, but whether the MEANING of my project shifted.*

**Today:** `git diff` shows you 47 files changed, 2000 lines added. Good luck figuring out what *actually* happened.

**Endgame:** `headspace diff HEAD~5` shows you:
```
Semantic drift detected:
  → Cluster "authentication" grew by 4 files (new OAuth2 provider)
  → Cluster "database" shrank — 2 files moved to "caching"
  → New cluster emerged: "rate limiting" (3 files, previously unclustered noise)
```

---

### 🧹 6. "Find the Rot"
> *As a maintainer, I want to find files that are semantically orphaned — they don't relate to anything else in my project.*

**Today:** Dead code, abandoned configs, half-written drafts — they accumulate silently.

**Endgame:** `headspace orphans` lists every file that HDBSCAN classified as noise — files that don't belong to any semantic cluster. These are your candidates for deletion, archival, or "wait, what IS this?"

---

## The Progression: Hack → Robust

The philosophy: **get the UX right with duct tape, then replace the duct tape with steel.**

| Phase | What | Hack (MVP) | Robust (Endgame) |
|-------|------|------------|-------------------|
| **1** | Find files by meaning | Web UI search box, cloud embeddings | OS-level hotkey, local embeddings, <200ms |
| **2** | See relationships | 2D PCA scatter plot in browser | Interactive graph with zoom, filter, neighborhood expansion |
| **3** | Know when things change | Full re-ingest every time | Content-hash change detection, incremental re-embedding |
| **4** | Reorganize intelligently | Show cluster labels in UI | CLI `suggest-layout` with dry-run + apply |
| **5** | Feed LLMs | Manual copy-paste | `headspace context` CLI with token budgeting |
| **6** | Find the rot | Noise points in cluster view | `headspace orphans` + `headspace duplicates` CLI |
| **7** | Ambient awareness | Manual "Ingest" button | Background daemon, filesystem watcher, always-on |
| **8** | Native experience | Axum web server in browser | Tauri desktop app / system tray / OS integration |

### The Rule
> **Every phase ships something usable. Every phase replaces something hacky from the last phase.**

Phase 1 is already done (the current MVP). Phase 3 (change detection) is the foundation that makes everything after it possible.

---

## What Makes This Different

There are file search tools. There are embedding databases. There are LLM context tools. Headspace is not any one of these — **it's the idea that your filesystem should understand itself.**

The closest analogy: **Git gave files a memory.** Headspace gives them **meaning.**
