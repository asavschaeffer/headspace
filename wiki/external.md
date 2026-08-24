# External knowledge

## Purpose

External knowledge layers connect a Headspace workspace to entities that live
outside it: encyclopedia articles, structured knowledge bases, and the general
web.

The external seam answers three questions: which outside sources a workspace is
willing to consult, how an outside entity is represented locally, and how
outside material may become local truth. It never answers identity questions on
its own — that remains the job of the [resolver](resolver.md) and
[Janus](janus.md).

## Decisions

### Source layers are per-workspace configuration

The source layers named in [Resolver](resolver.md) are not globally available.
Each workspace holds an ordered list of approved layers, and no external layer
is consulted unless it is enabled there. The default is all external layers
disabled: a fresh workspace asks nothing of the outside world.

```ts
type ExternalLayerId = string; // "wikipedia", "wikidata", "web"

interface WorkspaceExternalConfig {
  layers: { layer: ExternalLayerId; enabled: boolean }[]; // ordered by consultation priority
}
```

Order matters when several layers could answer the same question; the resolver
consults enabled layers in workspace order. Internal source layers (private,
project, shared, public corpora) are governed by [permissions](permissions.md)
and cross-corpus opt-in, not by this configuration.

```text
external layer absent   -> unknown to the workspace
external layer disabled -> never consulted
external layer enabled  -> consultable, in list order
default                 -> all disabled
```

### External entities are cached snapshots

An external entity is represented locally as a snapshot taken at a known time,
stored in the workspace — not as a chunk.

```ts
interface ExternalRef {
  layer: ExternalLayerId;  // which approved layer answered
  key: string;             // layer-scoped stable key, e.g. a Wikidata QID
  url?: string;            // human-followable address when one exists
  snapshotAt: string;      // when the payload was captured
  payload: unknown;        // layer-shaped snapshot content
}
```

Staleness is visible, not hidden: `snapshotAt` travels with the reference, and
anything rendering an external entity can show how old its knowledge is.
Refresh is explicit — the system never silently re-fetches and swaps a payload
underneath material that pointed at the old snapshot.

Because an `ExternalRef` is not a chunk, it has no revisions, no occurrences,
and no place in containment. It is workspace-cached evidence about the outside
world. This is also what separates it from a [binding](bindings.md): a binding
is a durable two-way correspondence with an external object the user edits;
an `ExternalRef` is a read-only snapshot of an entity the user consults.

### Links carry the relationship, in two strengths

A connection between local material and an external entity is a
[link](kernel.md) whose `toExternal` end is an `ExternalRef`. The role encodes
who asserted it:

```text
"candidate-referent" -> resolver-proposed; a possibility, not a claim
"references"         -> user-promoted; the author means this entity
```

Promotion from candidate to reference is an explicit user act, consistent with
the resolver's rule that mention is not reference. A candidate that is never
promoted stays a candidate; nothing external hardens into meaning by default.

### External content enters only by explicit copy

External payloads never leak into chunk content through resolution, rendering,
or caching. The only path from an external entity into a chunk is an explicit
`copy` operation, and that operation records full provenance: the source layer,
the key, and the `snapshotAt` of the snapshot copied.

```text
consult  -> ExternalRef cached in workspace, no chunk
link     -> "candidate-referent" or "references", no chunk content
ingest   -> explicit copy; new chunk; provenance records layer + key + snapshotAt
```

After ingestion the resulting chunk is ordinary local material with an
[operation-level provenance](provenance.md) trail naming its external origin.
The external actor form `external:<layer>` identifies the layer wherever its
facts enter the record.

## Working model

The first layers are `wikipedia` and `wikidata`: both defined, both disabled by
default. A generic `web` layer is specified for arbitrary URLs but comes later.

The initial implementation ships the types and the workspace cache store with
no live fetching at all. This is deliberate: candidate links, snapshot shapes,
staleness display, and the copy-with-provenance path can all be exercised
against manually seeded snapshots before any network code exists.

## Relationship to other seams

- [Resolver](resolver.md) consults enabled layers and packages results as
  `candidate-referent` links carrying evidence.
- [Janus](janus.md) guards the boundary between a name matching an external
  entity and the author meaning that entity.
- [Permissions](permissions.md) keep private material from leaking outward:
  consulting a layer sends queries, so enabling one is a visibility decision.
- [Provenance](provenance.md) records layer, key, and snapshot time whenever
  external material becomes local content.
- [Bindings](bindings.md) cover editable external objects; external knowledge
  covers consultable external entities. They do not overlap.
- [Views](views.md) render external entities in nebulas as a distinct source
  layer, visibly separate from local material.

## Open questions

- What does refresh do to existing links when a re-fetched snapshot differs
  materially from the one a `references` link was promoted against — is a
  changed external entity surfaced as a proposal, like a watched source?
- Does querying an enabled layer deserve its own outbound-privacy controls
  (batching, local-first candidate generation) beyond the enable switch?
- How are layer-scoped keys handled when an external source renames or merges
  its own entities (Wikidata redirects, moved URLs)?
