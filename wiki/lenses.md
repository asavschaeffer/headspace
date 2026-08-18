# Lenses

## Purpose

A lens is a named, saved, read-only query and projection over kernel facts and
indexes. It answers questions like "where does *very* appear, and who wrote
it?" or "which of this document's material came from a model?" and hands the
answer to a view as overlays, groupings, and highlights.

Lenses sit between the [Index](index.md) and [Views](views.md): the index
derives searchable knowledge, the lens selects and shapes it, and the view
renders it. None of the three touches truth.

## Decisions

### A lens is a saved query, not an operation

A lens has a stable identity so it can be reopened, but applying it changes
nothing in the kernel. Choosing a lens is view state, never a kernel commit
(see [Operations](operations.md)).

```ts
type LensId = string;

interface Lens {
  id: LensId;
  name: string;
  scope: { kind: "workspace" } | { kind: "container"; containerId: ChunkId };
  filter: LensFilter;
  groupBy?: "actor" | "actorKind" | "operationKind" | "date" | "term" | "role";
  renderHints: RenderHints; // e.g. coloring, highlighting, cluster overlays
}
```

Scope bounds the query: a workspace lens reads the whole workspace; a
container lens reads one container's contents.

### Filters are structured predicates, not free queries

A filter is built from a fixed predicate vocabulary. There is no free SQL, no
arbitrary code. Each predicate names the evidence it reads:

```ts
type LensFilter =
  | { kind: "term"; text: string }              // term index
  | { kind: "actor"; actorId: ActorId }         // provenance
  | { kind: "actorKind";
      value: "human" | "agent" | "driver" | "external" }
  | { kind: "date"; from?: string; to?: string }
  | { kind: "operationKind"; value: string }    // operation history
  | { kind: "echo"; span: SpanAddress }         // span echo index
  | { kind: "role"; value: string };            // link roles
```

```text
predicate       evidence source
--------------  -----------------------------------------
term            term index (word/icu@1 tokens)
actor / kind    revision createdBy, operation actorId
date            revision and operation timestamps
operationKind   operation records
echo            span echo index (normalized sentences)
role            durable links
```

The vocabulary is small on purpose. A predicate that cannot name its evidence
does not belong in a lens.

### Lens results are view material

A lens result illuminates a view — nebula overlays, groupings, term
highlights, provenance coloring. Manipulating a lens result never mutates
truth. Dismissing a grouping, recoloring a cluster, or closing the lens leaves
the kernel exactly as it was.

When a lens surfaces something worth keeping — an echo that turns out to be a
real motif, a similarity that deserves a name — promoting that finding into a
durable link is an explicit, separate operation by an authorized actor. This
is the settled [Index](index.md) rule: soft results stay soft until someone
promotes them.

### Lenses read only what the actor may see

A lens queries indexes, and indexes are scoped to one workspace with explicit
opt-in for anything shared ([Permissions](permissions.md)). A lens therefore
cannot leak private material through counts, groupings, or echoes: it projects
the evidence its actor is permitted to read, nothing more.

### Two lenses ship first

```text
term-search lens   term predicate over the term index; highlights and
                   groups occurrences of a term across the scope
provenance lens    actorKind grouping; colors human, model, driver, and
                   external material distinctly
```

Both are built in. Saved user-defined lenses use the same shapes.

## Working model

Canonical lenses the vocabulary is designed to express:

```text
term-by-author      "very" grouped by createdBy — who leans on the word
provenance clusters human vs model material via operation actors
style diagnostics   sentence-length distributions from the
                    decomposition cache (sent/icu@1)
motif echoes        span echo index surfacing "the lake is still"
                    wherever it recurs
```

Style diagnostics show why lenses read the decomposition cache and not just
durable chunks: sentence statistics need `sent/icu@1` parts without promoting
every sentence into the kernel ([Decomposition](decomposition.md)).

[Provenance](provenance.md) supplies the raw material for authorship lenses;
span-level authorship is derived on demand, consistent with operation-level
default authorship.

## Open questions

- How do predicates compose? Conjunction is clearly wanted ("*very*, by a
  model, this month"); whether the vocabulary needs disjunction and negation
  is undecided.
- Where do saved lens definitions live — per-workspace view state, or
  shareable objects a collaborator can open?
- May a lens regroup and therefore move stars in a nebula, or only illuminate
  them in place? [Views](views.md) leaves the illumination-versus-movement
  boundary open.
