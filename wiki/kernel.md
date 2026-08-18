# Kernel

## Purpose

The kernel defines the smallest truthful model shared by every Substrate
implementation. It owns chunks and the invariants required to manipulate them
safely. It should not know about filesystems, model providers, search engines,
databases, or user interfaces.

## Accepted principles

### Substrate truth is graph-shaped; default navigation is tree-shaped

The kernel should allow graph structure: chunks can reference, derive from,
transclude, and associate with many other chunks.

The user's default navigation should usually feel tree-shaped: documents,
folders, outlines, paragraphs, and local neighborhoods. The tree is a view or
projection over a graph, not the whole truth.

### A chunk is a continuing identity

A chunk is a stable address for an evolving conceptual object. Editing its
content does not make it a different chunk.

```ts
type ChunkId = string;

interface Chunk {
  id: ChunkId;
  currentRevisionId: RevisionId;
}
```

Chunk IDs should be globally unique and opaque. UUIDv7 is a plausible
implementation, not yet a required part of the model.

A chunk ID must not encode semantic or visual position. Identity must remain
stable while meaning and placement change.

Identity continuity is operational, not a semantic-similarity judgment:

```text
revise              -> same chunk, new immutable revision
copy/fork/branch    -> new chunk derived from a source revision
```

A revision may replace all prior content without changing chunk identity. The
sedimentary revision graph preserves what changed; governance determines who
may perform the operation. Questions about whether an external file was edited
or replaced belong to binding and conflict reconciliation, not to kernel
identity.

### A revision is an immutable historical state

Editing creates a revision rather than destroying the prior state.

```ts
type RevisionId = string;
type ContentHash = string;

interface Revision {
  id: RevisionId;
  chunkId: ChunkId;
  content: Content;
  contentHash: ContentHash;
  parentRevisionIds: RevisionId[];
  provenance: Provenance;
}
```

A content hash identifies exact content, not the continuing chunk. The
chunk's current revision is mutable state associated with its identity; it is
not part of the identity itself.

History and provenance are **sedimentary**: recorded automatically and
normally kept out of the user's way, but inspectable when curiosity,
debugging, recovery, or trust requires them.

### A content blob is not a chunk

A chunk is the continuing identity. A revision is a historical state of that
identity. A content blob is the immutable payload that the revision points at.

```ts
interface ContentBlob {
  hash: ContentHash;
  mediaType: string;
  bytes: Uint8Array;
}
```

Multiple chunks or revisions may point at the same blob if their content is
exactly identical. That is content interning, not shared chunk identity.

Same bytes may share storage. Same identity means the same continuing object.
Those are different claims.

### Content and arrangement are independent

Moving a chunk does not edit its content. Arrangement is represented by
relations rather than embedded into a content revision.

```ts
interface Relation {
  from: ChunkId;
  to: ChunkId;
  kind: RelationKind;
  position?: number;
}
```

Containment, ordering, reference, and derivation may be different relation
kinds. Their exact vocabulary remains open.

Containment should be acyclic. References and associative relations may form
cycles.

Containment does not require a chunk to have exactly one parent. Reuse and
multiple contexts are represented by multiple occurrences. Tree-shaped
navigation is a view over this structure, not a one-parent kernel invariant.

Arrangement changes participate in Git-style commit history. The store records
the resulting relation graph in the commit snapshot and attaches the operations
that explain how it changed.

### Placement, ancestry, and connection are distinct forms

The universal kernel recognizes three structural forms because they carry
different invariants:

- an **occurrence** places and orders material inside a container;
- a **derivation** records immutable ancestry from a source revision or span;
- a **link** records a durable explicit connection with an extensible role.

Containment through occurrences is acyclic. Derivation ancestry is acyclic and
sedimentary. Links may generally form cycles, while domain-specific roles and
constraints are interpreted above the kernel.

The three reuse verbs compose these forms:

```text
copy         -> create chunk + derivation
reference    -> create link with role "references"
transclusion -> create occurrence that resolves source content
```

Soft similarity remains an index result until a user or authorized operation
promotes it into an explicit link.

### Occurrences are appearances of chunks inside containers

The same chunk can appear in more than one place. The occurrence records a
particular appearance; the chunk records the underlying identity.

```ts
interface Occurrence {
  id: OccurrenceId;
  containerId: ChunkId;
  chunkId: ChunkId;
  position: Position;
  revision: "current" | RevisionId;
}
```

An occurrence can be moved, removed, pinned to a revision, or annotated without
changing every other occurrence of the same chunk.

### Product actions compose kernel primitives

Edit, move, reroll, contextualize, fork, and accept are meaningful user
intentions, but they are not necessarily kernel primitives.

A current candidate primitive set is:

```ts
create(chunk)
revise(chunkId, content, provenance)
relate(from, to, kind, position?)
unrelate(from, to, kind)
```

This is a working model, not an accepted final API. A generic
`manipulate(chunk, operation)` is disfavored because it hides invariants and
turns distinct state transitions into an unbounded switch.

## Current interpretation of actions

| Intention | Likely kernel effect |
| --- | --- |
| Edit | Create a revision and advance the current pointer |
| Move | Change arrangement relations without revising content |
| Contextualize | Read or traverse; often no kernel mutation |
| Reroll | Generate a proposal, then optionally revise or branch |
| Fork | Create a new chunk identity derived from a specific revision |
| Delete | Remove a relation or record a tombstone; unresolved |

Model output should be proposal-first. Generation should not silently
overwrite human work.

## Candidate invariants

- A `ChunkId` identifies one continuing object.
- A `Revision` is immutable and belongs to exactly one chunk.
- A chunk's current revision must belong to that chunk.
- Every revision records provenance.
- Default visible authorship is operation-level: a revision has a creator and
  source operations; span-level authorship is derived when needed.
- Content changes and arrangement changes remain distinguishable.
- Multi-step changes that must agree are applied atomically.
- Derived data such as embeddings cannot determine identity.

## Open questions

- Is deletion a tombstone, loss of containment, or both?

The minimum content representation is addressed in [Content](content.md): an
immutable byte blob with an explicit media type. Containment acyclicity and
multiple-parent behavior are accepted above.
