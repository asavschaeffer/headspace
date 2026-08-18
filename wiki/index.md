# Index

## Purpose

The index makes chunks discoverable and relatable. It derives knowledge from
kernel facts without becoming the authority for those facts.

The index is disposable in principle: every index can be rebuilt from the
[Store](store.md) and bound sources. Nothing an index holds is the only copy
of anything.

## Accepted principles

### Indexes derive; they never determine identity

An index entry is a computed fact about a revision. It can be dropped,
recomputed, or versioned without touching chunks, revisions, occurrences, or
relations. Identity and history live in the kernel and the store; the index
only makes them findable.

A soft relationship — a repeated phrase, a similar sentence, a shared term —
remains an index result until someone promotes it. Promoting a finding into a
durable link is an explicit, separate operation; querying an index never
mutates truth. [Lenses](lenses.md) follow the same rule: they read index
material and render it, and manipulating a lens result changes nothing in the
kernel.

### Entries are keyed by revision

Index entries are keyed by `revisionId`, not by chunk alone. Revisions are
immutable, so an entry computed against a revision stays correct forever for
that revision.

When a chunk is revised, the new revision is enqueued for reindexing and
staleness-sensitive query paths to the old revision are dropped. Staleness has
one definition everywhere: the entry's `revisionId` is no longer the chunk's
current revision.

### Private and public indexes are separate by default

Private content should index into private workspace indexes only.

Shared or public corpus indexes require explicit opt-in.

```text
private workspace -> private index
shared corpus     -> shared index
public corpus     -> public index
```

This avoids leaking private content through counts, first-seen claims,
occurrence search, similarity search, or deep-fate queries. See
[Permissions](permissions.md).

### Index instances are scoped to one workspace

Each index instance serves one workspace. Cross-corpus queries require
explicit opt-in; there is no ambient global index that quietly spans
workspaces. Permission-filtered global indexes remain possible later, but a
filtering mistake in such an index exposes private information, so scoping is
the foundation.

### Deletion and redaction reflow through the index

First-seen attribution and echo results are computed from currently visible,
permitted, non-redacted evidence, per [Deletion](deletion.md). Redacting or
tombstoning material does not corrupt the index; it changes what the index is
allowed to answer with. First-seen is a query result, not a permanent crown
stored as a fixed fact.

## The first index set

Four indexes exist first. All are derived, per-corpus, private by default,
and rebuildable from the store.

### Term index

An inverted index over `word/icu@1` tokens, normalized to NFC and lowercased,
mapping each token to its postings.

```ts
interface TermPosting {
  chunkId: ChunkId;
  revisionId: RevisionId;
  start: number; // UTF-16 code-unit offset into the revision's text
}
// token -> TermPosting[]
```

The term index powers text search and term lenses ("every occurrence of
'very', grouped by author").

### Interning index

Maps a blob hash to the revisions that carry it.

```ts
interface InterningEntry {
  blobHash: BlobHash;
  revisionIds: RevisionId[];
}
```

The interning index powers [Janus](janus.md) exact-equality: byte-identical
content is discovered here without collapsing chunk identity. First-seen is
answered as a query over this index restricted to visible, non-redacted
evidence — it is recomputed, not stored, so attribution can reflow when an
earlier source is redacted.

### Span echo index

Maps normalized sentence hashes to the spans that produced them. Sentences
come from `sent/icu@1`; normalization is NFC, lowercased, collapsed
whitespace.

```text
hash("the lake is still") -> spans in revision A, revision B, ...
```

The echo index powers "the lake is still" echoes and soft
[deep-fate](deep-fates.md) hints: repeated and near-repeated phrasing is
surfaced as derived fact, and stays derived until a user promotes a span or a
relation.

### Decomposition cache

Caches decomposer output by `(revisionId, method)` so that blocks, sentences,
and words of a revision are computed once.

```text
(revisionId, "sent/icu@1") -> DerivedPart[]
```

The cache is fully disposable; decomposers are pure functions over immutable
revisions, so any entry can be recomputed on demand. See
[Decomposition](decomposition.md).

## Deferred: embeddings and clustering

Embeddings are not in the first index set, but the design is retained. An
embedding describes the meaning of a particular revision. It does not identify
the chunk and does not directly determine its stable visual placement.

```ts
interface SemanticState {
  chunkId: ChunkId;
  revisionId: RevisionId;
  embedding: number[];
  model: string;
}
```

Staleness follows the general rule: a `SemanticState` is stale when its
`revisionId` is no longer the chunk's current revision. Similarity edges,
neighborhoods, and density clusters (such as HDBSCAN results) build on
embeddings and are deferred with them. An entity-candidate index is likewise
deferred to the [Resolver](resolver.md) and external-knowledge layers.

Other index products remain possible over time: metadata facets such as
author, date, project, and tags; ranked search; derived relation candidates.
Each follows the same contract as the first four.

## Responsibilities

- Derive searchable representations.
- Record which revision and algorithm produced derived data.
- Return evidence or scores with results.
- Tolerate rebuilding and incremental updates.
- Honor visibility: answer only from permitted, non-redacted evidence.

## Non-responsibilities

- Defining chunk identity.
- Holding the only copy of content or history.
- Deciding what context an agent receives.
- Owning final visual placement.
- Promoting soft findings into durable relations — that is always an explicit
  operation performed by a user or authorized actor.

## Open questions

- Which derived relations deserve promotion into durable human-authored
  relations, and should the system ever suggest promotion proactively?
- How do search and clustering preserve understandable locality once
  embeddings arrive?
