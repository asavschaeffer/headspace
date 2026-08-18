# Store

## Purpose

The store durably records kernel state and applies kernel changes atomically.
It answers **what Substrate knows**, not how an external file or service
represents it.

The store is a seam: an in-memory map, SQLite database, encrypted local store,
or synchronized remote store should be replaceable without changing kernel
meaning.

## Accepted model

### Commits snapshot authoritative graph state

Substrate uses a Git-like hybrid rather than full event sourcing.

Each immutable commit points to a structurally shared snapshot of chunk heads,
occurrences, and relations. It also cites the operations that produced that
state.

```ts
interface Commit {
  id: CommitId;
  parentIds: CommitId[];
  stateRootId: GraphStateId;
  operationIds: OperationId[];
  actorId: ActorId;
  createdAt: Timestamp;
}
```

The snapshot answers what is true at a commit. Operation records explain what
actors intended to do. Current state must not require replaying the complete
operation history.

Snapshots should use immutable structural sharing, so a commit reuses unchanged
objects rather than copying the entire workspace. A mutable workspace ref may
advance atomically to a new commit; committed state remains immutable.

Concurrent changes may create commits with the same parent. Reconciliation can
produce a later commit with multiple parents without destroying either history.

## Responsibilities

- Record chunks, immutable revisions, relations, and provenance.
- Resolve a chunk's current revision.
- Preserve sedimentary history.
- Apply related changes as one transaction.
- Detect conflicting attempts to advance the same state.
- Provide the facts needed for backup, recovery, and synchronization.

## Non-responsibilities

- Parsing files.
- Calling language models.
- Ranking search results.
- Choosing context.
- Computing screen coordinates.

## Working direction

SQLite is a plausible first durable implementation because it supports local
transactions, indexing, inspection, and migration without requiring the
kernel to become database-shaped.

The current `Map<string, Chunk>` conflates current state, content, and
history. Persistence should follow the kernel model rather than permanently
encode that temporary representation.

## Open questions

- Is the source of truth initially the filesystem, the store, or a carefully
  defined combination?
- What is the atomic unit of change: command, transaction, or event?
- How are concurrent edits detected and reconciled?
- Which history is permanent, compactable, or garbage-collectable?

