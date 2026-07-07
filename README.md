# AI-Embedded Folder Reorganization — Design Notes

*The hopper metaphor: redstone is signal, the piston does work on command, but the
hopper is what moves items between inventories **without a player standing there ticking
it**. The magic wasn't more powerful blocks — it was the connective tissue that let
blocks feed each other unattended. An LLM in a pipeline is a piston. The "AI-embedded"
magic is the hopper: **structured I/O and deterministic routing that lets the LLM's
output feed the next stage with no human in the loop.***

---

## Context is a budget you design, not a cost you pay

When an agent reads a file, the *whole thing* lands in the context window. Naively
surveying a folder is O(total bytes) — a single 200k-line log or a `node_modules` blows
the entire budget on noise. That's the amateur version.

The key realization: **how much of each file you read is a knob.** There's a tier ladder,
cheapest first, and you climb it only when you must:

1. **Metadata only — near-free, no content read.** Path, name, extension, size, mtime,
   is-it-binary, is-it-inside-a-git-repo. This alone classifies 60–70% of a dev folder
   correctly. You do *not* need to read a `.mp4` or a `Cargo.lock` to know what it is.
2. **Bounded peek.** For things that survive tier 1, read a *fixed token budget* — head +
   tail, first ~2KB. Critically: bounded **per file regardless of size**, so a giant file
   costs the same as a small one. A file's identity almost always lives in its first few
   lines (shebang, imports, headers, `# Title`).
3. **Full read — expensive, rare.** Reserve this for genuine ambiguity or for merge/split
   decisions where content actually matters.

And the biggest lever of all is **choosing the right atomic unit.** In a *dev* folder,
the unit isn't the file — it's the **project/repo**. You detect roots deterministically
(`.git`, `package.json`, `pyproject.toml`, `Cargo.toml`, …), treat each as an opaque
black box that moves as one unit, and *only recurse into the loose files that no project
claims.* That collapses "thousands of files" into "a few dozen units" before any LLM sees
anything. Clustering the individual files inside a repo is not just wasteful, it's
actively wrong — you'd shred functional units into semantic dust.

---

## The pipeline, re-shaped

The skeleton (report → embed → cluster → propose → visualize → nitpick →
dedupe/merge/split) is the right shape. The refinement everywhere: **shrink the LLM to
the smallest possible cog and make everything around it deterministic.** Here's the DAG
with the labor divided:

| Stage | Who does it | Why |
|---|---|---|
| Walk tree, detect project roots, hash files | **Deterministic** (code) | No intelligence needed; fast, reliable, free |
| Dedupe exact copies | **Deterministic** (content hash) | LLM would only add error and cost |
| Per-unit report (bounded peek → structured JSON) | **LLM, cheap model, fan-out** | The map step. One unit = one small independent call |
| Roll reports up the tree | **Deterministic reduce** | Summaries of summaries, not re-reading |
| Embed + cluster (HDBSCAN) | **Deterministic** | Statistical grouping |
| Propose reorg scheme | **LLM, smart model, once** | The one place taste/judgment pays off |
| Visualize, accept/reject per section | **Human** | The review gate |
| Merge / split file contents | **LLM, per-file, on demand** | Actual content transformation |
| Execute moves | **Deterministic, reversible** | `mv` with a replayable log |

The shape is **map-reduce / fan-out-fan-in**: N cheap independent LLM calls producing
structured fingerprints (embarrassingly parallel, and each one's context is *just that
unit* — they don't pollute each other), then a deterministic reduce, then *one* expensive
smart call that sees only the compact rolled-up summary, never the raw bytes. The
expensive model never reads a single file. That's the whole trick.

---

## Push-backs on the specifics

- **Reports, recursive & noisy — the failure mode is real.** The fix is the reduce: a
  directory's report is a *bounded summary of its children's reports*, not a
  concatenation. Fixed budget per node at every level, so noise can't compound as you go
  up. It's a tree of summaries, and each level is lossy on purpose.

- **Embed the reports, not the raw content — mostly.** Embedding reports is cheaper and
  *denoises* (the summarizer already threw away boilerplate). The cost: you're embedding
  the summarizer's *interpretation*, so if it flattens a real distinction, the cluster
  inherits that blindness. For a reorg, that's an acceptable trade — but keep hard
  metadata (language, project-membership, mtime) as explicit features *alongside* the
  embedding, not baked into it, so clustering can respect "these are the same project" as
  a hard constraint rather than hoping cosine similarity notices.

- **HDBSCAN — good pick, one caveat.** Right call over k-means because you don't have to
  guess the number of clusters and it can label things *noise* (which is honest — some
  files belong to no group). But clusters are **statistical**, and a folder's real
  structure is **functional/intentional**. "These three files are semantically similar" ≠
  "these belong in the same folder." Two configs can be near-identical embeddings and
  belong to two different projects. So clustering should *inform* the proposal, never *be*
  it — it's a hint the smart model reconciles against the hard project-boundary
  constraints, not the source of truth.

- **Dedupe is not an LLM job.** Exact dupes = content hash, deterministic, zero error.
  *Near*-dupes (v1 vs v2 of a script) — that's where the LLM earns its keep, because "are
  these the same thing?" needs judgment. Split the two; don't spend a model on the part
  arithmetic solves.

- **Accept/reject per section + visualize — this is the actual product.** The reorg is a
  **diff over the filesystem**, and the whole UX is reviewing a diff: expand a subtree,
  approve this bucket, reject that move, edit a destination. The non-negotiable
  underneath it: **every move is reversible and partial-apply-able** — a replayable
  `old → new` log so "put that part back" is one command, not archaeology. That
  reversibility is what makes it safe enough to actually run.

---

## The meta-point

Once the LLM emits *structured, schema-conformant* output instead of prose, it stops
being a chatbot and becomes a **component** — something a `for` loop can call a thousand
times and a downstream stage can trust without reading. That's the hopper. The
prose-in-prose-out chat interface is the piston you hand-activate; schemas + routing are
what close the loop into automation. That single shift — **LLM as typed function, not
conversation** — is most of what "AI-embedded" buys you.
