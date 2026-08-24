# The Janus problem

## Problem

Two pieces of text can be byte-identical while being different authorial
objects. They look in two directions at once:

- content equality says they are the same bytes;
- identity and provenance may say they are different events, memories, or
  objects.

Example: someone wrote `the lake is still` two years ago, and another writer
writes `the lake is still` today.

Headspace should be able to reveal the deep relation between those phrases
without forcing them to be the same editable object.

Janus watches over thresholds: same/different, copy/edit, source/descendant,
identity/equality, and permission/provenance.

## Accepted distinction

Identical text may share immutable content storage and may link to the first
known occurrence or ancestor. It should not automatically share mutable chunk
identity.

```ts
interface Blob {
  hash: BlobHash;
  mediaType: string;
  text: string;
}

interface InterningEntry {
  hash: BlobHash;
  revisionIds: RevisionId[]; // every non-redacted revision carrying this payload
}
```

Interning is a derived index fact ([Index](index.md)); first-seen attribution
is computed from it on demand, never stored as a crown.

The content address is SHA-256 over `mediaType + "\0" + text`, so the
interning key is the media type plus the payload, not raw bytes alone.

This gives the desired deep fate:

- exact duplicates point at the same content blob;
- later duplicates can point back to the first-seen revision;
- each authorial object keeps its own chunk identity;
- editing one object creates a new revision and breaks exact equality only for
  the changed region.

Janus is responsible for protecting this distinction. It prevents an exact
content match from accidentally becoming edit authority over another object.

## Granularity

The phrase `the lake is still` can match at multiple levels:

- whole phrase;
- words: `the`, `lake`, `is`, `still`;
- spans: `the lake is`, `lake is still`;
- semantic variants: `the lake is stormy`.

These matches should not all become durable chunks by default. Most are derived
index facts. A subspan becomes a durable chunk only when the user promotes it by
editing, linking, annotating, moving, or otherwise treating it as an object.

## Edit behavior

If a user changes `the lake is still` to `the lake is stormy`, the system can
preserve several truths at once:

- the phrase chunk receives a new revision;
- the old revision still points at the old content blob;
- the unchanged derived span `the lake is` can still match prior text;
- `stormy` does not need to become a separate durable chunk unless promoted;
- the index records that the new phrase shares a prefix/span with the old
  phrase.

The connection breaks at the changed span, not necessarily at the whole
document.

## Implementation direction

Use three layers:

1. content-addressed storage for exact immutable payloads;
2. derived span indexes for repeated phrases and subphrases;
3. explicit relations for promoted authorial objects.

The exact-match layer is deterministic and safe. The span layer is powerful but
can become large, so it should be indexed on demand and bounded by useful
granularities. The promoted-object layer is the durable kernel truth.

## Non-goals

Do not make all identical text the same chunk. That would make editing,
authorship, provenance, and context unsafe.

Do not materialize every token, word, and phrase as a durable kernel object by
default. That creates overwhelming state and makes ordinary writing feel like
database maintenance.
