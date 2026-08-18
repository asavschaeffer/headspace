# Index

## Purpose

The index makes chunks discoverable and relatable. It derives knowledge from
kernel facts without becoming the authority for those facts.

The index is disposable in principle: it should be possible to rebuild it
from the store and bound sources.

## Accepted principles

### Private and public indexes are separate by default

Private content should index into private workspace indexes only.

Shared or public corpus indexes require explicit opt-in.

```text
private workspace -> private index
shared corpus     -> shared index
public corpus     -> public index
```

This avoids leaking private content through counts, first-seen claims,
occurrence search, similarity search, or deep-fate queries.

## Possible index products

- Lexical terms and ranked text search.
- Metadata facets such as author, date, project, and tags.
- Embeddings associated with specific revisions.
- Similarity edges or neighborhoods.
- Density clusters such as HDBSCAN results.

## Accepted distinction

An embedding describes the meaning of a particular revision. It does not
identify the chunk and does not directly determine its stable visual
placement.

```ts
interface SemanticState {
  chunkId: ChunkId;
  revisionId: RevisionId;
  embedding: number[];
  model: string;
}
```

## Responsibilities

- Derive searchable representations.
- Record which revision and algorithm produced derived data.
- Return evidence or scores with results.
- Tolerate rebuilding and incremental updates.

## Non-responsibilities

- Defining chunk identity.
- Holding the only copy of content or history.
- Deciding what context an agent receives.
- Owning final visual placement.

## Open questions

- Is a soft chain stored as a relation or computed as an index result?
- Which derived relations deserve promotion into durable human-authored
  relations?
- How should stale embeddings be handled after revision?
- How do search and clustering preserve understandable locality?
