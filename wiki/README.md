# Headspace architecture wiki

This directory is a working notebook for deriving Headspace's architecture
from first principles.

These pages are not a specification yet. They distinguish:

- **Accepted principles**: conclusions established in discussion.
- **Working model**: the smallest design that currently satisfies those
  principles.
- **Open questions**: decisions that still need to be reasoned through.

The [question ledger](questions.md) orders unresolved decisions by dependency
and records which question is currently under discussion. A recommendation in
that ledger is not an accepted principle until the discussion resolves it.

Only stable, compressed conclusions should eventually move into
`headspace-brief.md`.

## Current map

- [Kernel](kernel.md): identity, revisions, relations, and invariants.
- [Content](content.md): immutable payloads, media types, and text layers.
- [The Janus problem](janus.md): identical text, shared storage, provenance,
  and identity.
- [Deep fates](deep-fates.md): where material came from, where it went, and
  how it transformed.
- [Provenance](provenance.md): operation-level authorship, source history, and
  accountability.
- [Permissions](permissions.md): authority for view, reuse, transclusion, and
  edit operations.
- [Deletion and redaction](deletion.md): sever, tombstone, unpublish, redact,
  and hard-delete semantics.
- [Store](store.md): durable recording and transactions.
- [Adapters](adapters.md): replaceable capability implementations, with source
  and projection adapters worked through in detail.
- [Bindings](bindings.md): correspondence with external objects.
- [Decomposition](decomposition.md): derived parts on demand, durable identity
  on commitment.
- [Index](index.md): discovery, similarity, and derived knowledge.
- [Lenses](lenses.md): saved, read-only queries and projections over facts and
  indexes.
- [Resolver](resolver.md): possible referents for spans, names, concepts, and
  ambiguous mentions.
- [External knowledge](external.md): approved outside layers, local snapshots,
  and explicit ingestion.
- [Operations](operations.md): select, reduce, generate, and user intentions.
- [Proposals](proposals.md): suggested change recorded inert until accepted.
- [Conflicts and merge](conflicts.md): divergent truths and their explicit,
  lossless resolution.
- [Collaboration](collaboration.md): actors, roles, and cross-boundary flows.
- [Views](views.md): semantic position, visual placement, and spatial memory.
- [Plan](plan.md): dependency-ordered milestones and the proof each one owes.
- [Question ledger](questions.md): ordered design backlog and decision status.
- [Information shapes](information-shapes.md): architectural fixtures for chats, articles, and future formats.
