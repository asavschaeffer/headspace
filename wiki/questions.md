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

### Deletion semantics

> When a user deletes a star, which thing are they asking Substrate to remove?

A visible star may be a chunk, occurrence, derived part, transclusion, link, or
view placement. The kernel already distinguishes severing an appearance from
tombstoning an identity, but the product needs safe defaults that match what the
user believes they selected.

## Queued

### Kernel

1. What does deletion do to identity, containment, history, and blobs?

### Persistence and external truth

1. During the first product phase, which facts are authoritative in the
   filesystem and which are authoritative in the Substrate store?
2. What is the atomic unit of change, and is an event log required alongside
   current state?
3. How are concurrent internal and external edits detected and reconciled?
4. Which sedimentary history may be compacted or garbage-collected?
5. What cardinalities and historical behavior do bindings permit?

### Interpretation and operations

1. Is decomposition a driver policy or a separate replaceable seam?
2. How are chunk identities preserved across repeated imports?
3. How does a driver declare and expose lossy projection?
4. Are select, reduce, and generate universal operations or one important
   pipeline over more fundamental reads and writes?
5. Should `generate` become a broader transformation concept that includes
   human editing?
6. What is the lifecycle of a proposal?

### Index and views

1. When is a soft relation merely an index result, and when is it promoted to
   a durable relation?
2. How are stale derived representations handled after revision?
3. Which parts of visual placement are durable user state and which are
   reproducible layout output?
4. What anchors a stable home layout while search, filtering, and clustering
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
