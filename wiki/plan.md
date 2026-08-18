# Implementation plan

## Purpose

This page orders the build. Every other page says what Substrate is; this one
says what gets built first, what proof each stage owes, and — honestly — how
much of it exists right now. The plan is dependency-ordered: nothing here
overrides a design page, and a milestone is not complete until its named proof
passes.

## Decisions

### Every milestone names its proof

A milestone is a claim plus a test that would falsify it. "Kernel done" means
the invariant tests pass; "persistence done" means a reloaded store is
indistinguishable from the one that wrote the log. Progress is measured by
proofs, not by lines.

### The kernel is built before anything that depends on it

The order runs inward-out: types and invariants first, then durability, then
operations, then the seams that translate the world in and out ([drivers](drivers.md)),
then the derived layers ([index](index.md), [lenses](lenses.md)) that are
rebuildable and therefore safe to build last. Derived layers never determine
identity, so deferring them costs nothing but features.

### Milestones

```text
milestone               contents                                proof
M1  kernel              types + transactions                    invariant tests: revision
                                                                immutability, occurrence
                                                                acyclicity, atomic promote,
                                                                proposal freshness
M2  persistence         .substrate log/blobs/snapshot           crash-safe append; reload
                        behind StorePort                        equivalence:
                                                                materialize(log) === snapshot
M3  operations +        full transaction vocabulary;            generation is proposal-first
    proposals           select/reduce/generate pipeline         end to end
M4  markdown driver     ingest -> sidecar -> project ->         round-trip: import then export
                        external-edit reconcile                 is byte-stable on untouched
                                                                files
M5  decomposition +     decomposers + promote shapes +          selection-to-promotion works
    promotion           UI selection-to-promotion               from the star surface
M6  index + search      term, interning, and span echo          nebula search runs on the
                        indexes                                 term index
M7  lenses v1           term lens + provenance lens             both lenses render in the
                                                                nebula
M8  resolver +          candidate types + cache store,          types compile; cache stores
    external stubs      no live fetching                        and returns snapshots
```

M1 covers the shared kernel vocabulary — `Chunk`, `Revision`, `Blob`,
`Occurrence`, `Link`, `Derivation`, `Operation`, `Proposal` — and the
transactions of [Operations](operations.md), each applied atomically. Its
proof suite pins the invariants of [Kernel](kernel.md): a `Revision` is never
mutated after creation; occurrence containment admits no cycle; `promote` is
one commit however many objects it touches; `accept` refuses a proposal whose
target's current revision is no longer in `basisRevisionIds`.

M2 implements the `StorePort` seam of [Store](store.md) — `openWorkspace`,
`append(commit)`, `readLog(from)`, `loadSnapshot`, `saveSnapshot`, `getBlob`,
`putBlob` — over the `.substrate/` layout. The reload-equivalence proof is the
central one:

```ts
// After any sequence of commits, restarting must lose nothing:
deepEqual(materialize(readLog()), loadSnapshot().state);
```

M3 makes model output real without making it authoritative: `generate`
records `inputRevisionIds` from the reduced set and always yields a
[Proposal](proposals.md), inert until accepted.

M4 is the first driver and the first authority split: the store authoritative
for identity, structure, history, and provenance; the bound file authoritative
for bytes edited outside Substrate; divergence reconciled through the driver
per [Drivers](drivers.md) and [Conflicts](conflicts.md), never silently.

M5 delivers the three promotion shapes of [Decomposition](decomposition.md)
— extracted chunk, copied chunk, addressable span — reachable from a text
selection in the star.

M6 builds the first index set of [Index](index.md); M7 puts the first two
[lenses](lenses.md) on the nebula; M8 ships the types and cache store of
[External knowledge](external.md) and the [resolver](resolver.md) candidate
shapes with fetching disabled.

### Deferred deliberately

Later stages, in no committed order: the SQLite `StorePort` backend, a Tauri
desktop shell, embeddings and clustering (`SemanticState` is designed;
[views](views.md) already treat soft position as derived), the
[collaboration](collaboration.md) transport and sync layer, and the redaction
UI over the semantics fixed in [Deletion and redaction](deletion.md). Each has
a designed seam waiting for it; none blocks the milestones above.

## Status

No milestone is complete. What exists is a working prototype that proves the
product loop — navigate the nebula, focus a star, select, reduce — and the
seam layout, not the kernel:

- The star and nebula surfaces render and navigate a chunk tree.
- `select` and `reduce` exist as pure functions over an in-memory store;
  `generate` is not wired, and nothing is proposal-shaped yet.
- A one-way filesystem ingest reflects a folder into a snapshot of chunks
  bound to source paths — import only, with no sidecar, no projection, and no
  reconciliation.
- Search is naive substring match, not the term index.
- The prototype's chunk carries mutable text and a version counter rather
  than immutable revisions; there are no occurrences, links, derivations,
  operations, or proposals, and no `.substrate/` store — state is a JSON
  snapshot loaded into a `Map`.

The prototype's value is that every stub sits where a milestone will replace
it: its store is the `StorePort` socket, its ingest script is the markdown
driver's socket, its search function is the term index's socket. The immediate
front is M1 — replacing the prototype chunk with the kernel vocabulary — and
everything after follows the order above.

## Open questions

- Whether M4 and M5 can swap order: promotion is more visible in the product,
  but reconciliation risk is retired earlier if the driver lands first.
- What snapshot cadence M2 should use (every N commits, on idle, on close) —
  the reload-equivalence proof holds under any of them.
