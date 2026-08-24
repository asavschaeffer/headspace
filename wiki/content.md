# Content

## Purpose

Content is the durable payload a revision points at. It is not the whole chunk,
and it is not the whole user-visible object. Identity, history, placement,
provenance, indexes, and view layout live around content.

## Accepted principles

### Blob means immutable payload

A content blob is the stored payload for a revision.

```ts
interface Blob {
  hash: BlobHash;
  mediaType: string;
  text: string; // v1 payloads are text; binary media arrives behind the same seam
}
```

For the first implementation, the safest default is an exact immutable payload
plus an explicit media type. Examples:

- `text/markdown; charset=utf-8`
- `text/plain; charset=utf-8`
- `image/png`
- `application/json`

The hash is over the canonical stored payload for that media type. Concretely,
the content address is SHA-256 over `mediaType + "\0" + text`, so the
interning key is the media type plus the payload, not raw bytes alone.

### Blob equality is not identity equality

Two chunks can have different identities while pointing at the same content
blob. This is useful deduplication, not a semantic claim that they are the same
authorial object.

Editing one chunk creates a new revision pointing at a new blob. Other chunks
that pointed at the old blob are unchanged.

### Text interpretation is layered

For text, there are several possible layers:

- bytes: exact stored UTF-8 payload
- decoded text: Unicode string after decoding bytes
- normalized text: optional comparison/indexing form
- parsed structure: Markdown blocks, paragraphs, sentences, tokens
- rendered view: what the user sees

The kernel should not collapse these into one thing. Exact bytes preserve file
round-tripping. Normalized forms and parsed structures can be derived by
adapters, decomposers, or indexes.

### Different content types can share one interface

The kernel-level interface should support text, images, structured documents,
and generated composites without forcing them into one text model.

The kernel only needs to know that a revision points at durable content with a
media type. Specialized behavior belongs behind seams:

- adapters import/export external formats
- decomposers expose smaller derived parts on demand
- indexes build searchable or semantic projections
- views render and manipulate user-facing forms

## Working model

Plain text and Markdown chunks can start as byte blobs. Composed documents can
be represented structurally by occurrences and relations, then exported to
Markdown or another file format by an adapter.

This means a document may be:

1. a simple blob, when it is just imported text;
2. a structure of child occurrences, once parts are promoted;
3. both, with a source blob plus derived structure, during reconciliation.

The product should keep that distinction invisible unless the user is debugging
history, provenance, or synchronization.
