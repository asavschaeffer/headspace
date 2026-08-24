# Headspace — Axiom Distillation

This is the evolving kernel contract for Headspace: the small set of decisions
from which the data model, operations, permissions, and views should follow.
It records settled principles, not speculative implementation detail.

## Identity, history, and deletion

**A1. Identity and provenance are durable by default.**

Chunks are continuing identities. Revisions are immutable historical states.
Content blobs are immutable payloads, and content equality does not establish
identity equality.

**A2. Ordinary delete changes a local composition, not global existence.**

Removing material from a document omits its occurrence from the document's next
revision. It does not delete the source chunk, revisions, blob, or provenance.

**A3. Historical truth is append-only; visibility and availability are
policy-governed.**

The graph records that an operation or relationship occurred. Policies may
control who can see it and whether its material remains available.

**A4. Redaction, retirement, and purge are distinct, explicitly authorized
operations.**

Redaction replaces visible material and restricts prior material. Retirement
ends a chunk's live use while preserving a tombstone. Purge is exceptional
erasure for legal, privacy, or security reasons; it is never the ordinary
delete path.

## Branching and continuity

**A5. A chunk has one authoritative current revision.**

An essay may accumulate revisions (`essay@1 → essay@2 → … → essay@11`) without
creating additional stars or cluttering the Nebula. This is its sediment.

**A6. Divergent authorial futures create distinct chunk identities.**

A deliberate fork mints a new star only when material has become substantively
a different thing—for example, `essay-poem` rather than another essay draft.

**A7. Cross-identity influence belongs to provenance, not identity continuity.**

The new chunk is related to its source revision through `forked_from` and the
fork operation. Ordinary revision parents describe continuity of the same
chunk; they do not collapse separate chunks into one identity.

## Filesystem persistence and reconciliation

**A8. Files are editable projections; graph truth is recorded in the kernel.**

An external file edit is input to a reconciliation operation. A driver may
interpret it, but does not become an independent truth seam.

**A9. Promoted content carries durable, invisible anchors by default.**

Managed Markdown serializes anchors for promoted chunks and occurrences, where
identity must survive external editing. Ordinary prose, unpromoted spans, and
derived structure remain anchor-free.

**A10. Reconciliation may propose identity correspondence but must not silently
resolve Janus ambiguity.**

The driver matches explicit anchors first and may use similarity for ordinary
re-entry. Exact duplication, convergent text, fork-versus-revision, and
split-or-merge ambiguity require a Janus decision rather than a guess.

**A11. Clean export strips Headspace anchors.**

Anchors are internal bookkeeping, not an external interchange format. Files
that leave Headspace may return without them; reconciliation then follows A10.

## Open questions

1. Promotion and decomposition
2. Operation lifecycle
3. Lenses
4. View architecture
5. Transaction invariants
6. Trust and abuse
