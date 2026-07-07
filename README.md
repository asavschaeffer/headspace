# The Chunk — `syscall(0)`

> The kernel object. Everything — hopper, headspace, the loom-weave — is userland built
> from these. Nail this one object's identity, provenance, and mutability rules and the
> rest is composition. Nothing else needs inventing.

The whole design lives in one idea: **a chunk has four zones, each with a different
mutability rule.** Confuse two of them and the system rots (attribution lies, forks
corrupt their origins, drag-to-reorder scrambles history). Keep them separate and every
hard operation — edit, rearrange, fork, re-ingest — falls out for free.

| Zone | Mutability | Answers |
|---|---|---|
| **Identity** | immutable for life | "who am I" — survives edit & rearrange |
| **Content** | versioned (content-addressed) | "what do I say right now" |
| **View** | freely mutable | "where did *you* put me" |
| **Provenance** | append-only | "where did I come from, who touched me" |

---

## The object

```ts
type ChunkId = string;   // ULID — minted once, stable for the chunk's entire life
type Hash    = string;   // content address; changes when content changes

interface Chunk {
  // ── IDENTITY (immutable) ───────────────────────────────────────────────
  id: ChunkId;
  causal_seq: string;      // logical clock at birth — the "what actually happened" order

  // ── CONTENT (versioned) ────────────────────────────────────────────────
  kind:
    | 'message' | 'heading_section' | 'paragraph'
    | 'list' | 'list_item' | 'code_block' | 'quote';   // = markdown AST node role
  text: string | null;     // leaves carry text; containers carry structure (null here)
  content_hash: Hash;      // leaf: hash(text). container: Merkle over ordered child hashes

  // ── VIEW / curated arrangement (mutable) ───────────────────────────────
  parent_id: ChunkId | null;   // null = root (a message or an ingested document)
  order_key: string;           // fractional index among siblings; drag-to-reorder writes here

  // ── PROVENANCE (append-only) ───────────────────────────────────────────
  origin: {
    actor: 'human' | { model: string };
    at: string;             // timestamp
    source_id: string;      // the message / conversation / document it was born in
  };
  derived_from?: { id: ChunkId; op: 'fork' | 'split' | 'merge' };
  edits: Array<{ at: string; actor: 'human' | { model: string }; from: Hash; to: Hash }>;
}
```

*(Rust struct is a mechanical translation: immutable fields, `edits: Vec<Edit>` grow-only,
the four zones map cleanly onto ownership. TS is canonical because the UI lives there.)*

---

## The three non-obvious calls, and why

**1. `id` ≠ `content_hash`.** Identity is a ULID minted once; content is hashed
separately. Editing the text changes `content_hash`, **not** `id`. This is the inode/bytes
split: the chunk is the same *thing* before and after you rewrite it, so comments,
provenance, and inbound links stay attached across an edit. If identity *were* the hash,
every keystroke would orphan everything pointing at the chunk.

**2. Fork is copy-on-write, never a shared reference.** `fork(selection)` deep-copies the
selected sub-forest; each copy gets a **new `id`** and a `derived_from: { id, op: 'fork' }`
edge back to its origin. The fork is independent from birth — editing it can never mutate
the source — but the lineage is a hard link you can always walk backward. A shared-id fork
would be a footgun: curate the branch, corrupt the trunk.

**3. `order_key` is fractional, and it lives apart from `causal_seq`.** This is the *two
orderings* made concrete. `causal_seq` is immutable birth order — the truth, what you audit
and attribute by. `order_key` is a fractional index (LexoRank-style) you can rewrite to
drop a chunk between any two siblings without renumbering the rest — the curated view,
what you *read* and what continuation feeds the model. They diverge the instant you
rearrange, and that's *correct*: the system holds both, always.

---

## How it survives the four operations

Every op is a function that touches exactly one or two zones and preserves the rest:

| Operation | Mutates | Preserves | Mechanism |
|---|---|---|---|
| **edit** | Content, Provenance | Identity, View | new `content_hash`; append `{from,to}` to `edits` |
| **rearrange** | View only | everything else | rewrite `parent_id` / `order_key`; `causal_seq` untouched |
| **fork** | — (creates new) | origin untouched | COW copy; new `id` + `derived_from` edge |
| **re-ingest** | — (indexes) | everything | idempotent on `(id, content_hash)`; Merkle skips unchanged subtrees |

The Merkle hashing on containers is what makes re-ingest cheap: tomorrow's index only
re-embeds the subtrees whose root hash moved. Rearranging a list doesn't change any leaf's
text, so nothing re-embeds — only the view updates. Editing one item bubbles a new hash up
its ancestors and *only that spine* re-indexes. This is hopper's "bounded reduce," now
serving change-detection.

And the attribution question you cared about — *"did I think this, or did the model?"* — is
answered by construction: `origin.actor` says who bore the chunk, and the `edits` log says
who changed it and how. The archive can never quietly launder a model's words into yours.

---

## The syscall table

The four mutations above, plus the primitive we found underneath fork / edit-continue /
hopper-reduce:

```
select(predicate)      -> Chunk[]        // by content, by kind, by subtree, by search hit
reduce(Chunk[], budget) -> Context       // linearize by order_key, respect nesting, cap tokens
generate(Context)      -> Chunk[]        // attach to a new tree (fork) or this one (continue)

edit(id, text)         -> Chunk
rearrange(id, parent, order_key) -> Chunk
fork(Chunk[])          -> Tree
reingest(Tree)         -> Index
```

That's the interface. **hopper** = `select` the file tree, `reduce`, propose. **headspace**
= `reingest` your corpus, `select` by search, roll up to parents. **the loom-weave** =
`select` a section, `reduce`, `generate` into a new tree (fork) or this one (edit-continue).
Three products, zero new primitives.

Build this object and its seven calls, make one of them real end-to-end, and the OS exists —
small, boring, and load-bearing. Everything grand is downstream of that.

---

## `syscall(1)` — the first heartbeat (shipped)

`generate` is wired. It is not in the kernel — it's a **driver** (`src/model-driver.mjs`),
because `generate` translates *thought* the way an ingestion driver translates a file. It is
a pure seam: `complete(messages) -> string`. Any OpenAI-compatible endpoint fills it —
we run **NVIDIA NIM** (`meta/llama-3.1-70b-instruct`) and **OpenRouter** free tiers today;
swapping to Claude later is a one-line change and the kernel never notices.

The verb decomposes exactly as promised:

```
gather(store, targetId)      // SELECT  — target + its lineage + its siblings (the whole category)
assemble(store, chunks)      // REDUCE  — a bounded, *readable* context manifest + token estimate
generate(store, {targetId})  // GENERATE— context -> model -> reply parsed back into chunks,
                             //           threaded into the tree with model provenance +
                             //           `context:` = the manifest it was made from (it remembers)
```

Run it live: `node heartbeat.mjs` (needs `NVIDIA_NIM_API_KEY` in env). It ingests the triage
message, selects `mog`, and generates a real reply — model-authored chunks knit into the tree.

Files: `src/model-driver.mjs` (the driver + seam), `heartbeat.mjs` (live proof),
`generated-seeds.json` (captured real replies), `substrate-home.html` (the home — the demo
grown into a place, with the full select→reduce→generate choreography).

---

## `syscall(2)` — the store, made durable (shipped)

*Memory that forgets isn't memory.* `src/store-disk.mjs` persists the whole store to disk and
brings it back — **content-addressed**: a chunk's text lives in a blob keyed by its hash, so
identical content is stored exactly once (real dedup). On load, the Merkle root is recomputed
from scratch, so a matching root hash is *proof* of integrity, not trust.

```
saveStore(store, file)   // -> { chunks, blobs, bytes }   the floor drivers pour into
loadStore(file)          // -> Store, clock resumed, Merkle verified
```

Proven (`node persist-demo.mjs`): a tree round-trips with an identical Merkle root; 4 identical
lines collapse to 2 blobs; and the **real Projects workshop** — 2064 chunks from the fs-driver —
persists to **1.16 MB** and reloads intact (8 GB of projects held as a 1.16 MB *map*, because we
store shape, not bulk). The seam to a Rust/SQLite backend is a straight swap of these two
functions — the kernel never learns which is behind it (Joseph's chapter, when this becomes a
Tauri desktop home).

Now durable, the bottom half of the wishlist is buildable: content sub-drivers (pdf/docx/code +
secret-safety) can deposit permanently, and a semantic index finally has something that persists
to work on.
