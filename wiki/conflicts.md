# Conflicts and merge

## Purpose

Conflicts govern what happens when one chunk accumulates two truths: two
revisions descending from the same parent, or a store state and an external
file state that no longer agree. Merge is how those truths become one again
without losing either.

The rule underneath everything on this page: **never silent loss**. Whichever
path a divergence takes, both sides survive in history and the resolution is
an explicit, attributable commit.

## Decisions

### A conflict is structural, not emotional

A chunk's revisions form a DAG through `parentRevisionIds`. A conflict exists
in exactly two situations:

```text
two heads        -> two revisions descend from one parent and neither
                    is an ancestor of the other
store vs external -> a driver observes that a bound file's bytes diverged
                     from what the store last projected
```

Nothing else is a conflict. In particular, a watched transclusion whose source
moved on is not a conflict — the local occurrence is pinned, so the local
document is never in a contested state (see [Deep fates](deep-fates.md)).

### Merge mechanics for leaf text

Leaf text merges by diff3 at line level against the common ancestor revision.

```text
non-overlapping hunks -> auto-combined into a "merge" proposal
overlapping hunks     -> conflict proposal carrying both sides verbatim
```

A clean merge is still a proposal: a merge combines two authored sides, and
combined content — like model output and collaborator suggestions — never
applies itself ([Proposals](proposals.md)). The driver's fast path is not a
merge; it applies only when one side changed (see [Drivers](drivers.md)). A
personal setting may auto-accept clean merges; the accept is still recorded as
that actor's operation.

A conflicted merge does not invent a resolution. The proposal carries both
sides' hunks verbatim, and resolving it is authorship: the accepting actor
chooses or composes the surviving text.

### Merge mechanics for composites

A composite revision is a list of child occurrences, so composites merge at
the occurrence level: blocks added, removed, and moved on each side are
compared against the common ancestor's occurrence list. The same split
applies — non-conflicting occurrence changes combine into a clean merge
proposal; both sides touching the same occurrence produces a conflict
proposal. Child chunks whose own content diverged merge separately as leaf
text; a composite merge never rewrites a child's content.

### Every merge is a two-parent commit

An accepted merge produces one revision whose `parentRevisionIds` names both
heads, in one atomic commit ([Operations](operations.md), [Store](store.md)).
The DAG closes; neither head is deleted or rewritten. History keeps the fork,
the two heads, and the join — sedimentary, inspectable, out of the way.

```ts
// after accepting a merge of heads a and b:
const merged: Revision = {
  id: newRevisionId,
  chunkId,
  blobHash: mergedBlobHash,
  mediaType,
  parentRevisionIds: [a, b],   // the join is visible in the DAG
  createdBy: acceptingActorId,
  createdAt,
  operationId,                 // the accept's operation
};
```

The merge proposal itself is an ordinary `Proposal` with
`kind: "merge"` and `basisRevisionIds` naming the heads it was computed
against. The freshness rule from [Operations](operations.md) applies
unchanged: if either head advances before accept, the proposal flips to
superseded and a fresh merge may be offered against the new heads.

### Case map

```text
case                          resolution
----------------------------  ------------------------------------------------
internal + external file      reconciliation proposal from the driver; block
edits to the same doc         matching by hash, then order + similarity
                              (see Drivers)
watched source changed        source-update proposal; never a conflict — the
                              local occurrence stays pinned until accept
two users edit one chunk      same revision DAG, same merge path; future
                              collaboration changes the transport, not the
                              model
upstream source redacted      no merge occurs; attribution reflows at query
or severed                    time, and a severed or redacted transclusion
                              source renders as a tombstone placeholder
markdown sidecar lost         identity reconstructed by content-hash matching;
                              unmatched blocks become new chunks with a
                              reconciliation note
```

### Store-vs-external divergence is reconciliation, not overwrite

The store is authoritative for identity, structure, history, and provenance;
a bound file is authoritative for bytes the user edits outside Substrate
([Bindings](bindings.md)). When they diverge, neither side silently wins.

The [Drivers](drivers.md) reconcile path decides the shape: if only the
external side changed, matched changes apply directly as `revise` by
`driver:fs`; if both sides changed, the divergence becomes a
`"reconciliation"` proposal and resolves through the same accept path as any
merge. Either way the external state and the internal state both enter
history before anything is combined.

### Losing the sidecar loses identity honestly

If a driver's round-trip memory is destroyed, chunk identity for that file
must be reconstructed from evidence: blocks are re-matched by content hash
against known revisions. Exact matches rebind to their existing chunks.
Unmatched blocks become new chunks, and the reconciliation records a note
saying so.

Identity loss is possible in this case, and it is **reported, not hidden**. A
block that was edited externally while the sidecar was gone cannot prove its
lineage; Substrate says that plainly rather than guessing an ancestry it
cannot support.

### Redaction never merges

Redaction and tombstoning downstream of reuse are not merge cases. There is
nothing to combine: attribution reflow is computed at query time from visible,
non-redacted evidence ([Deletion](deletion.md)), and documents that
transcluded a now-severed or redacted source render a tombstone placeholder in
its position. The downstream document's own revisions are untouched.

## Open questions

- Should conflicted-hunk presentation offer word- or sentence-granular
  resolution built on the decomposition methods, or is line-level diff3
  sufficient indefinitely?
- How does the auto-accept-clean-merges setting interact with workspace roles
  once multiple humans share a workspace — per-actor, per-workspace, or
  gated by role?
- N-way divergence (three or more heads) currently resolves as successive
  two-parent merges; whether a single n-parent merge commit is ever worth the
  complexity is unsettled.
