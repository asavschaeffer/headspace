# Substrate — a primer

**How to read this.** Each level below is the *entire* system, told truthfully, at a higher
resolution than the one before it. Level 0 is one sentence. Level 4 is the whole machine with
its reasons. Stop wherever it stops being useful; go deeper only where you're curious. Nothing
in a lower level contradicts a higher one — it just adds detail. (This shape — a summary of
summaries, each layer lossy on purpose — is also literally how the system itself handles
information. You'll learn the idea by reading something built out of it.)

A **glossary** at the very bottom defines the jargon. When a sentence loses you, the noun you
don't know is probably there.

---

## Level 0 — one sentence

Every idea and every file becomes the same kind of object — a **chunk** in a tree — that you
can search, rearrange, and hand to an AI, whose reply comes back as more chunks in the same tree.

---

## Level 1 — one paragraph

The system is made of **five parts** and does **three things**. The five parts (the *anatomy*):
a **kernel** that defines the one universal object, a **driver** layer that turns the outside
world into those objects, an **index** that makes them findable, a **binding** that ties an
object back to the real thing it came from, and a **store** that remembers them on disk. The
three things (the *physiology*): **select** (gather a subset of objects), **reduce** (flatten
that subset into a compact context), and **generate** (send that context to an AI and turn its
reply back into objects). Everything the system does is one of those three verbs, aimed
somewhere. Everything it *is* is one of those five nouns.

The mantra: **kernel · driver · index · binding · store** / **select · reduce · generate.**

---

## Level 2 — the five parts and the three motions

### The object they all share: the chunk
A **chunk** is one node of a document or conversation — a paragraph, a list item, a heading and
the section under it, or a whole file. Chunks link into a **tree** (a heading contains its
bullets; a folder contains its files). The tree *is* the structure of the thing, parsed. This is
the one idea everything rests on: **one object type the whole system is built from** — the way a
Unix system makes everything "a file."

### The five parts (anatomy — what it's made of)

- **kernel** — defines the chunk, the tree, and the handful of operations. Pure logic, no
  reading or writing of the outside world. Small and stable; you build it once.
- **driver** — turns some messy external thing into chunks. A markdown driver parses text; a
  filesystem driver walks a folder; a *model* driver turns a context into an AI reply. Drivers
  are where all the mess of the real world is quarantined, so the kernel stays clean.
- **index** — makes chunks findable and relatable: by word (search) or, eventually, by meaning
  (embeddings) and by group (clustering).
- **binding** — a pointer on a chunk back to the real thing it represents (a file path, a URL).
  It's what lets "open it" or "work on it" mean something.
- **store** — where chunks live so they survive you closing the program. On disk, encrypted,
  with identical content saved only once.

### The three motions (physiology — what it does)

- **select** — gather a subset of chunks. "This chunk, plus its parents, plus its siblings." A
  search is a select. Everything starts here.
- **reduce** — take that subset and flatten it into one bounded block of text (a *context*),
  small enough to hand off. Summaries of summaries, never the raw pile.
- **generate** — send that context to an AI and parse its reply back into chunks, attached to
  the tree. The reply is data of the same kind as everything else — not a special "chat message."

That's the whole loop: **select → reduce → generate.** Search, fork a conversation, "help me
think about this note," reorganize a folder — all of them are these three verbs in some order.

---

## Level 3 — how it actually works

### The chunk has four zones, each with a different rule
This is the most important mechanical detail. A chunk's fields split into four groups, and each
group is allowed to change in a different way:

1. **identity** (`id`, `causal_seq`) — set once, *never changes*. Editing a chunk's text does not
   change its id. So links to it survive edits. (Think: a file's inode vs. its bytes.)
2. **content** (`kind`, `text`, `content_hash`) — *versioned*. The hash is a fingerprint of the
   text; for a container it's a fingerprint built from its children's fingerprints (a **Merkle**
   hash), so a branch's fingerprint changes only if something inside it changed.
3. **view** (`parent_id`, `order_key`) — *freely movable*. This is where the chunk sits and in
   what order among its siblings. Rearranging touches only this.
4. **provenance** (`origin`, `derived_from`, `edits`) — *append-only*. A growing log of who made
   this and who changed it (human vs. AI), never rewritten. This is how the system can always
   answer "did I write this, or did the model?"

Keep those four rules straight and every hard operation becomes easy. Confuse two of them and the
system rots (rearranging would scramble history; an edit would orphan every link).

### Two clocks, on purpose
Because *view* is movable but *identity* is fixed, a chunk has **two orderings at once**:
`causal_seq` (the true order it was created — what actually happened) and `order_key` (the order
*you* arranged it in — what you want to read). They start equal and drift apart when you
rearrange. The system keeps both: one is the truth for auditing, the other is the view for
reading. Neither overwrites the other.

### How the three verbs actually run
- **select** = `gather(store, targetId)`: walk the tree to collect the target chunk, its
  ancestors, and its siblings. Pure tree-walking, no AI.
- **reduce** = `assemble(chunks)`: sort them, flatten to text with a size cap, and produce a
  **manifest** (the list of which chunks went in) so you can see exactly what will be sent.
- **generate** = `generate(store, {targetId})`: build the context, run it through the **redact
  gate** (scrub any secrets), send it to the model driver's `complete()`, then `parse()` the
  markdown reply into new chunks and attach them — tagged as authored by the model, remembering
  the manifest they were built from.

### Drivers in, drivers out
"Ingesting" your files is just a driver (**world → chunks**). It's tiered by cost: read the
cheap deterministic things first (name, size, dates, a content-hash for finding exact
duplicates, a short peek at the first lines), and treat a whole project (a `.git`/`package.json`
folder) as **one** chunk instead of shredding it into thousands. The expensive, private tier —
summarizing or embedding a file's contents — is drawn as a hard line: **the API line.**
Everything below it is free, offline, private; only above it does anything leave your machine.
A **default-deny** rule means file bodies aren't read at all unless you explicitly allow it, and
the redact gate scrubs credentials before anything is stored or sent.

---

## Level 4 — why each decision, and the operating-system lens

### The design choices, and the reason for each
- **id ≠ content_hash.** Identity is a permanent handle; content is a changing value. Splitting
  them is what lets you edit a chunk without breaking everything pointing at it.
- **Merkle hashing.** By fingerprinting containers from their children, you get near-free
  change-detection: re-saving or re-indexing only has to touch the branches whose fingerprint
  moved. Rearranging changes no text, so nothing re-indexes.
- **Copy-on-write fork.** "Fork this section into a new tree" makes *new* chunks that link back
  to the originals (`derived_from`), rather than sharing them. So editing the fork can never
  corrupt the source, but the lineage is always traceable.
- **Two orderings.** Explained above — the price of letting you rearrange freely while keeping an
  honest record of what actually happened.
- **The LLM as a typed function, not a chatbot.** The model is called with a structured context
  and returns structured chunks. Because its output is data (not prose in a chat window), a
  `for` loop can call it a thousand times and the next stage can trust the result without a human
  reading it. *That* shift — from conversation to component — is most of what makes this
  "AI-embedded" instead of "a chat box."
- **Deterministic around a tiny LLM core.** Wherever plain code can do the job (walking, hashing,
  deduping, moving), use code — it's free, fast, and never wrong. Spend the model only where
  judgment is actually required. The expensive model should see a small rolled-up summary, never
  the raw bytes.

### The operating-system lens (the mental model that makes it click)
The system is shaped like an OS, and the mapping is exact, not poetic:

| substrate | operating system |
|---|---|
| **chunk** — one universal object | "everything is a file" |
| **id / provenance** — a pointer back to origin | addressing (every byte has an address) |
| **select → reduce → generate** — assemble a context and run it | **exec** — assemble a program's state and run it |
| **fork / continue** — branch off a chunk selection | **`fork()`** — spawn a child process from a parent's state |
| **index** — how chunks find each other | the discovery/lookup layer |
| **binding** — chunk → real file | a handle to a device/resource |
| **store** — durable chunks | the disk |

A normal OS addresses **files and processes**. This one addresses **thoughts and their lineage**.
That's the whole reason it feels like a new kind of operating system rather than a note app: it's
a new bottom layer — a filesystem whose unit is an idea, that remembers who made each one, and can
grow new ones on request.

---

## Glossary — terms to look up

- **chunk** — the one object type: a node of a document/conversation/file, in a tree.
- **kernel** — the small core that defines the object and the operations; no I/O, no dependencies.
- **driver** — code that translates one kind of external thing (a file, a folder, an AI call)
  into chunks. Polyglot by nature; where the mess lives.
- **seam** — a stable interface you can swap implementations behind (e.g. any AI endpoint fits the
  same `complete(messages) → text` shape). "You have nothing but seams" = every part is swappable.
- **content-addressed** — stored/looked-up by a hash of the content itself, so identical content is
  automatically stored once (that's how the store dedupes).
- **hash / fingerprint** — a short deterministic code computed from data; same data → same code,
  different data → (almost certainly) different code.
- **Merkle (hash tree)** — a hash of a container built from the hashes of its children, so any
  change deep inside bubbles up and is detectable at the top cheaply.
- **provenance** — the record of where something came from and who touched it. Here: append-only.
- **fractional index (`order_key`)** — a number/string you can always insert *between* two others
  without renumbering the rest; how drag-to-reorder works.
- **select / reduce / generate** — the three operations. Gather; flatten-to-context; run-the-model.
- **context** — the bounded block of text handed to the model for one call. Its **manifest** is the
  list of which chunks went into it (so you can see exactly what's being sent).
- **fan-out / fan-in (map-reduce)** — do many small independent jobs in parallel (fan-out), then
  combine their results (fan-in). Used to summarize each unit separately, then roll up.
- **embedding** — a list of numbers representing a piece of text's *meaning*, so "similar meaning"
  becomes "close together," enabling search-by-meaning. (Not built yet — the "index" gap.)
- **clustering / HDBSCAN** — grouping items by similarity without being told how many groups exist;
  can also label some items "noise" (belonging to no group). Statistical, not authoritative.
- **idempotent** — an operation you can run twice with no extra effect (re-ingesting an unchanged
  chunk does nothing). Makes syncing safe.
- **default-deny** — the safe posture: read/send *nothing* unless explicitly allowed, rather than
  read everything and try to exclude the dangerous bits.
- **the API line** — the boundary between free/offline/private work and work that costs money and
  leaves your machine (summarizing, embedding). Enforced by the redact gate.
- **redact gate** — the checkpoint that scans content for secrets (keys, passwords) and scrubs them
  before anything is stored or sent.
- **exec / fork** (OS terms) — *exec*: load a program's state and start running it. *fork*: make a
  child process that's a copy of a parent's state. Used here as the exact analogy for
  generate and for branching a conversation.
