# Substrate architecture wiki

This directory is a working notebook for deriving Substrate from first
principles.

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
- [Drivers](drivers.md): translation between external forms and chunks.
- [Bindings](bindings.md): correspondence with external objects.
- [Index](index.md): discovery, similarity, and derived knowledge.
- [Resolver](resolver.md): possible referents for spans, names, concepts, and
  ambiguous mentions.
- [Operations](operations.md): select, reduce, generate, and user intentions.
- [Views](views.md): semantic position, visual placement, and spatial memory.
- [Question ledger](questions.md): ordered design backlog and decision status.
- [Information shapes](information-shapes.md): architectural fixtures for chats, articles, and future formats.
