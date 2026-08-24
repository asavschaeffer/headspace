# Store

## Purpose

The store durably records kernel state and applies kernel changes atomically.
It answers **what the workspace graph records**, not how an external file or
service represents it.

The store is a seam: the file-backed workspace store, a SQLite database, an
encrypted local store, or a synchronized remote store should be replaceable
without changing kernel meaning.

## Accepted model

### Authority is split at the workspace boundary

The workspace store is authoritative for identity, structure, history, and
provenance. A bound external file is authoritative for bytes the user edits
with external tools. Divergence between the two is reconciled through the
adapter; neither side silently wins (see [Adapters](adapters.md) and
[Conflicts](conflicts.md)). Within the workspace, the store is simply the truth.

A filesystem path never carries identity; correspondence to external objects
is recorded by [Bindings](bindings.md), and the store keeps the facts that
make reconciliation possible.

### Commits snapshot authoritative graph state

The store uses a Git-like hybrid rather than full event sourcing.

Each immutable commit records the operation that was performed and the kernel
facts that resulted from it: new revisions, occurrences, links, derivations,
proposals, tombstones.

```ts
interface Commit {
  id: CommitId;
  parentIds: CommitId[];
  at: string;
  actorId: ActorId;
  operation: Operation;   // intent: what the actor did and against what inputs
  facts: CommitFacts;     // result: the kernel objects this commit made true
}

interface CommitFacts {
  revisions?: Revision[];
  occurrences?: Occurrence[];
  links?: Link[];
  derivations?: Derivation[];
  proposals?: Proposal[];
  tombstonedChunkIds?: ChunkId[];
}
```

The materialized snapshot answers what is true. The operation preserves what
the actor intended to do. Current state never requires replaying the complete
history: a periodic snapshot plus the log tail materialize it.

A mutable workspace head may advance atomically to a new commit; committed
state remains immutable.

Concurrent changes may create commits with the same parent. Reconciliation
produces a later commit with multiple parents without destroying either
history (see [Conflicts](conflicts.md)).

### The atomic unit of change is one operation

One operation = one commit = one log append. Multi-object changes — a
`promote` that extracts a chunk, rewrites the parent as a composite, and
records the derivation — are a single commit. Related changes that must agree
are never split across commits.

The operation vocabulary itself belongs to [Operations](operations.md); the
store only guarantees that each commit applies atomically or not at all.

### Physical layout

The first durable backend is pure TypeScript with no native dependencies,
rooted at the workspace:

```text
.headspace/
  log.jsonl        append-only committed transactions
  blobs/ab/abcd…   content-addressed immutable payloads (sha-256 hex, 2-char fan-out)
  snapshot.json    periodic materialized state + log offset (rewritten atomically via temp+rename)
  sidecars/…       adapter-owned round-trip memory (see adapters)
```

`log.jsonl` is the sequence of commits. Blobs are immutable and shared:
revisions with identical content point at one payload (interning, per
[Content](content.md) and [Janus](janus.md)). `snapshot.json` is a
convenience materialization, always reconstructible from the log; it is
rewritten atomically so a crash leaves either the old snapshot or the new
one, never a torn file. Sidecars belong to [Adapters](adapters.md) and hold no
kernel truth.

### Hashing

Blob hashes are SHA-256 over the UTF-8 bytes of `mediaType + "\0" + text`.
The implementation is pure TypeScript, synchronous, and verified against the
FIPS 180-4 test vectors. Payloads are text in this phase; the hash identifies
exact content, never continuing identity ([Kernel](kernel.md)).

### Concurrency

One workspace has one writer, enforced by a lock file. The commit DAG —
`parentIds` is plural by design — is the seam where future concurrent
writers and sync merges plug in; divergent heads resolve through merge
proposals rather than a CRDT commitment (see [Conflicts](conflicts.md) and
[Collaboration](collaboration.md)).

### History is sedimentary and kept by default

Revisions, resolved proposals, and superseded state remain in the log.
Compaction and blob garbage collection — dropping blobs unreachable from any
revision — are explicit administrative operations, never automatic.
Redaction is a governance operation with its own semantics
([Deletion](deletion.md)), not a storage optimization.

## The store port

Backends sit behind one interface; the seam is the port, not SQL, and the
kernel is not redesigned for any backend.

```ts
interface StorePort {
  openWorkspace(root: string): Promise<Workspace>;
  append(commit: Commit): Promise<void>;
  readLog(from?: CommitId): AsyncIterable<Commit>;
  loadSnapshot(): Promise<Snapshot | undefined>;
  saveSnapshot(snapshot: Snapshot): Promise<void>;
  getBlob(hash: BlobHash): Promise<Blob | undefined>;
  putBlob(blob: Blob): Promise<void>;
}
```

SQLite is the planned second backend behind the same port. It offers local
transactions, indexing, inspection, and migration without requiring the
kernel to become database-shaped.

## Responsibilities

- Record chunks, immutable revisions, relations, and provenance.
- Resolve a chunk's current revision.
- Preserve sedimentary history.
- Apply related changes as one atomic commit.
- Detect conflicting attempts to advance the same state.
- Provide the facts needed for backup, recovery, and synchronization.

## Non-responsibilities

- Parsing files.
- Calling language models.
- Ranking search results.
- Choosing context.
- Computing screen coordinates.

## Open questions

- Snapshot cadence: what triggers a snapshot rewrite — commit count, log
  size, idle time, or explicit request?
- Log segmentation: whether and how `log.jsonl` rotates or segments as a
  workspace grows.
- Encryption at rest for the file-backed layout.
