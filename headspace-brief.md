# Headspace brief

## The idea

Headspace is a spatial environment for navigating, editing, and connecting writing, projects, files, and AI conversations.
> **Headspace is a spatial interface over a versioned workspace graph of human- and machine-created information, held together by a tiny composable kernel.**

Its foundation is one common object, a small set of explicit boundaries, and reviewable operations.

## The experience

Headspace has two primary surfaces.

### The Star

A Star is the focused product surface for one addressable piece or scope. It can focus something as broad as a directory or as small as a derived text span. Its canvas can act as a text editor without making Star itself a kernel concept.

The user can:

- Write and edit
- Reveal adjacent or related chunks
- Search the wider workspace
- Attach and connect material
- Rearrange or fork sections
- Select context
- Dispatch an agent
- Accept, edit, branch, or discard its proposal

Agents operate on explicit targets and context. Their results return as reviewable proposals inside the Star surface.

The trick is to understand that up from afar a directory looks like a star. but diving into it, even one document can house a nebula.

### The Nebula

The Nebula turns indexes, placements, and relations into a beautiful, meaningful spatial map.

Stars chain together hard and soft.
- hard chain: tokens in a sentence, or paragraphs in an essay, back and forth of a chat, or contents of a project
- soft chain: HBDscan embedding similarity clustering, provenance (by: {person}, imported from: {photos}/{desktop}, date: {range(a,b)}, projects, tags


I am considering dileniating constellations and nebulas but i want constellations to be optionally breakable or at least flexible to be able to recluster all the stars during search or via filters or even just from breathing animation

Search illuminates relevant stars and clusters without removing them from their surroundings. A stable home layout preserves spatial memory.

Clicking a cluster opens it. Clicking a star moves into focused work. There will be ux decisions on how to open something or dive deeper into its elements though i think it all could be the same button. If we imagine the star interface's background to be it's constellation, and behind that the nebula it should be technically possible. we will have to make some compromises between beauty and functionality and I am willing to incorporate 'typical' file explorer elements like indexes.


## The model

Everything promoted into durable identity becomes a **chunk**: an addressable unit such as a paragraph, section, message, file, or directory. Chunks form a graph; surfaces may project that graph as a tree when tree-shaped navigation is useful.

A chunk separates:

- Stable identity
- Versioned content
- Mutable arrangement
- Recorded authorship and history

Versioning and provenance are supporting guarantees, not the product itself. They ensure edits, references, and human–model collaboration remain trustworthy.

## The architecture

The implementation separates:

- **Kernel:** the workspace graph vocabulary, transactions, and invariants
- **Client:** the interactive browser session and its optimistic local state
- **Host:** the authoritative local runtime that owns filesystem access and external credentials
- **Store:** the append-only history and durable payloads
- **Index:** disposable, rebuildable discovery and relation projections
- **Surfaces:** Nebula for spatial navigation and Star for focused work
- **Seams and adapters:** capability boundaries and their replaceable implementations, including ingestion, projection, collaboration, indexing, and persistence

The three operations are:

- **Select:** choose relevant material
- **Reduce:** compile it into a bounded representation
- **Generate:** ask a collaborator to transform it into an inert proposal

Direct human editing remains an ordinary kernel transaction; it does not need to masquerade as model generation.

Everything is a seam. Filesystems, model providers, indexes, stores, interfaces, and parsers must remain replaceable. The kernel should know as little as possible about them.

```text
world → adapters → chunk kernel → store
                      │
                 index + bindings

select → reduce → generate → new chunks
```

## The starting product

The first version is a lean web application for displaying and working with personal writing, projects, and imported LLM conversations, with selective collaboration.

It should prove one loop:

```text
navigate → focus → compose → dispatch → integrate → return
```

The existing filesystem remains authoritative. Headspace reflects its contents and binds representations in the workspace graph back to their sources.

## The direction

If the workspace proves valuable:

1. Build a high-performance, encrypted, local-first desktop application.
2. Add filesystem watching, offline models, synchronization, and native editing.
3. Allow Headspace's versioned workspace graph to become authoritative for human knowledge.
4. Expose ordinary filesystem paths as a compatibility interface.
5. Explore deeper operating-system integration only after conventional files become the limiting abstraction.
