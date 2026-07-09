# Headspace brief

## The idea

Substrate is a spatial workspace for navigating, editing, and connecting writing, projects, files, and AI conversations.
> **Substrate is a spatial interface for an evolving tree of human and machine-created information, held together by a tiny composable kernel.**

Its foundation is one common object, five replaceable parts, and three universal operations.**

## The experience

Substrate has two primary surfaces.

### The Star

The star is the discrete kernel unit. It's upper bound is a directory, but it can be as small as a single token. The UX is a simple canvas that can act as a text editor.

The user can:

- Write and edit
- Reveal adjacent or related chunks
- Search the wider workspace
- Attach and connect material
- Rearrange or fork sections
- Select context
- Dispatch an agent
- Accept, edit, branch, or discard its proposal

Agents operate on explicit targets and context. Their results return, of course, as stars.

The trick is to understand that up from afar a directory looks like a star. but diving into it, even one document can house a nebula.

### The Nebula

The nebula replaces the index with a beautiful, meaningful spatial map.

Stars chain together hard and soft.
- hard chain: tokens in a sentence, or paragraphs in an essay, back and forth of a chat, or contents of a project
- soft chain: HBDscan embedding similarity clustering, provenance (by: {person}, imported from: {photos}/{desktop}, date: {range(a,b)}, projects, tags


I am considering dileniating constellations and nebulas but i want constellations to be optionally breakable or at least flexible to be able to recluster all the stars during search or via filters or even just from breathing animation

Search illuminates relevant stars and clusters without removing them from their surroundings. A stable home layout preserves spatial memory.

Clicking a cluster opens it. Clicking a star moves into focused work. There will be ux decisions on how to open something or dive deeper into its elements though i think it all could be the same button. If we imagine the star interface's background to be it's constellation, and behind that the nebula it should be technically possible. we will have to make some compromises between beauty and functionality and I am willing to incorporate 'typical' file explorer elements like indexes.


## The model

Everything meaningful becomes a **chunk**: an addressable unit such as a paragraph, section, message, file, or directory. Chunks connect into trees.

A chunk separates:

- Stable identity
- Versioned content
- Mutable arrangement
- Recorded authorship and history

Versioning and provenance are supporting guarantees, not the product itself. They ensure edits, references, and human–model collaboration remain trustworthy.

## The architecture

The five parts are:

- **Kernel:** chunks, trees, and invariants
- **Driver:** translates external things into or out of chunks
- **Index:** makes chunks discoverable and relatable
- **Binding:** connects chunks to real files, URLs, or other objects
- **Store:** persists content and history

The three operations are:

- **Select:** choose relevant material
- **Reduce:** compile it into a bounded representation
- **Generate:** transform it through a model<?? what about just human editing lol>

Everything is a seam. Filesystems, model providers, indexes, stores, interfaces, and parsers must remain replaceable. The kernel should know as little as possible about them.

```text
world → drivers → chunk kernel → store
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

The existing filesystem remains authoritative. Substrate reflects its contents and binds representations back to their sources.

## The direction

If the workspace proves valuable:

1. Build a high-performance, encrypted, local-first desktop application.
2. Add filesystem watching, offline models, synchronization, and native editing.
3. Allow Substrate’s versioned content store to become authoritative for human knowledge.
4. Expose ordinary filesystem paths as a compatibility interface.
5. Explore deeper operating-system integration only after conventional files become the limiting abstraction.