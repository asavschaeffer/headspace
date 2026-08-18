# Question ledger

## Purpose

This ledger orders unresolved design questions so the wiki can grow into a
specification without pretending that every plausible recommendation is an
accepted decision.

Statuses mean:

- **Current**: the question presently being discussed.
- **Queued**: unresolved and ordered behind prerequisite questions.
- **Accepted**: resolved and distilled into the relevant topic page.

The topic pages remain authoritative for accepted principles. This page is the
route through the discussion, not a second specification.

## Current

### Placement durability

> Which parts of visual placement are durable user state, and which are
> reproducible layout output?

Spatial memory is a product promise: search, clustering, and corpus growth
must not arbitrarily destroy where things were. Whether a placement is saved
state, an anchored derivation, or a pure projection decides what the view
layer may persist and what it must always be able to recompute
([Views](views.md)).

## Queued

### Views

1. What anchors a stable home layout while search, filtering, and clustering
   change what is emphasized?

## Accepted and distilled

- Kernel truth is graph-shaped; default navigation is tree-shaped.
- Containment is acyclic, but the kernel does not impose exactly one parent.
- References and associative relations may form cycles.
- The minimum kernel content is an immutable byte blob with an explicit media
  type; parsing and normalization are layered interpretations.
- Reuse distinguishes copy, reference, and transclusion; authored documents
  prefer watched transclusion.
- Attention does not promote derived material into durable identity;
  commitment does.
- Editing advances the same chunk to a new immutable revision regardless of the
  magnitude of the content change.
- Copy, fork, and branch create new chunk identities derived from a source
  revision; governance controls authority, while provenance records the action.
- Arrangement and relation changes participate in Git-style sedimentary
  history.
- Immutable, structurally shared commit snapshots are authoritative state;
  attached operation records preserve intention and provenance without making
  full event replay necessary.
- The universal structural forms are occurrence, derivation, and link:
  placement, ancestry, and explicit connection remain distinct because they
  obey different invariants.
- An LLM conversation is canonically a message graph with revision-specific
  reply links; readable transcripts and named branches are paths through it.
- Granularity is progressive and demand-driven. Views may render derived
  sections, sentences, words, or subword tokens as stars without promoting them
  into durable chunk identities.
- Deleting a star severs the occurrence the user is looking at; tombstoning
  the identity everywhere is an explicit, distinct action ([Deletion](deletion.md)).
- The store is authoritative for identity, structure, history, and provenance;
  a bound file is authoritative for bytes edited outside Substrate
  ([Store](store.md)).
- The atomic unit of change is one operation, one commit, one log append, in
  an append-only log with periodic snapshots ([Store](store.md)).
- Concurrent edits are detected by a single-writer lock and the plural-parent
  commit DAG; divergence resolves through merge proposals ([Store](store.md),
  [Conflicts](conflicts.md)).
- History is sedimentary and kept by default; compaction and blob garbage
  collection are explicit administrative operations ([Store](store.md)).
- A binding targets a chunk; one chunk may carry many bindings, one file binds
  one doc chunk, renamed files are rediscovered by content hash, and binding
  history is sedimentary ([Bindings](bindings.md)).
- Decomposition is a registered, versioned method behind its own seam, not
  driver-private policy; a driver chooses which method applies to its format
  ([Decomposition](decomposition.md), [Drivers](drivers.md)).
- Chunk identity survives repeated imports through driver sidecar memory,
  reconstructed by content-hash matching when the sidecar is lost
  ([Drivers](drivers.md)).
- Lossy projection is declared explicitly; opaque blocks pass through
  byte-stable rather than being silently dropped ([Drivers](drivers.md)).
- Select, reduce, and generate are a pipeline over the transaction vocabulary,
  not kernel primitives; generation terminates in `propose`
  ([Operations](operations.md)).
- Human edits in one's own workspace apply directly as `revise`; generation
  does not widen to absorb human editing ([Operations](operations.md)).
- A proposal is an inert record moving open to accepted, rejected, or
  superseded, with basis freshness validated at accept
  ([Operations](operations.md), [Proposals](proposals.md)).
- Every operation is one atomic transaction and one commit
  ([Operations](operations.md), [Store](store.md)).
- A soft relation remains an index result until explicitly promoted into a
  durable link ([Index](index.md)).
- Staleness has one definition: a derived entry whose revision is no longer
  the chunk's current revision; stale entries are recomputed, never trusted
  ([Index](index.md)).
