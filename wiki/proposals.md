# Proposals

## Purpose

A proposal is a suggested change that has been recorded but not applied. It is
how Substrate holds work that originated outside the local author's own hands —
model output, upstream source changes, collaborator suggestions, detected
relations, driver reconciliations, merges — without letting any of it silently
become truth.

This page owns the semantics and taxonomy of proposals. The formal lifecycle
and freshness rules live on the [Operations](operations.md) page.

## Decisions

### One concept unifies external change

Every path by which change arrives from outside the author's direct editing is
the same kernel object: a Proposal record, inert until accepted.

```text
kind                origin
----                ------
generation          model output from the select/reduce/generate pipeline
source-update       a watched transclusion's source published a new revision
suggested-edit      another actor suggests a change to material they cannot edit
detected-relation   an index finding offered for promotion into a durable link
merge               two revision heads of one chunk combined against a common ancestor
reconciliation      a bound external file and the store both changed
```

Human edits in one's own workspace apply directly as `revise`. Everything
else — model output, upstream watched-source updates, collaborator
suggestions, detected relations, merges, driver reconciliations when both
sides changed — is always a proposal. This is the general form of the settled
rule that generation is proposal-first: no actor's output overwrites an
author's work without the author's explicit acceptance.

An external-only file edit is the exception: with no internal edits since the
last sync it applies directly as `revise` by `driver:fs`, because it is the
author's own editing arriving through the driver ([Drivers](drivers.md)).

### A proposal is an inert record

```ts
interface Proposal {
  id: ProposalId;
  kind: "generation" | "source-update" | "suggested-edit" | "detected-relation" | "merge" | "reconciliation";
  status: "open" | "accepted" | "rejected" | "superseded";
  basisRevisionIds: RevisionId[];   // what the proposal was computed against
  targetChunkIds: ChunkId[];
  payload: ProposedChange[];        // inert operation descriptions, applied only on accept
  createdBy: ActorId; createdAt: string;
  resolution?: { by: ActorId; at: string; operationId?: OperationId };
}
```

The payload is a description of operations, not the operations themselves.
Creating a proposal creates no revisions, occurrences, links, or derivations on
its targets. Only acceptance applies the payload, atomically, as one commit.

The basis records which revisions the proposal was computed against. This makes
every proposal an honest claim: "given these exact states, here is a suggested
change." Whether the claim is still applicable is a freshness question governed
by the lifecycle rules in [Operations](operations.md), not a property frozen
into the record.

Provenance is complete on both ends. `createdBy` records who or what proposed;
`resolution` records who resolved it, when, and — for acceptance — the
`operationId` of the commit that applied the payload. A proposal therefore
carries its own deep fate: from the reduced inputs that produced it, through
the actor that offered it, to the operation that made it real.

### Lifecycle is defined once, on the operations page

```text
open -> accepted | rejected | superseded
```

The [Operations](operations.md) page is the single source of truth for the
lifecycle: how `accept` validates basis freshness, when a stale accept is
refused and the proposal flips to superseded, and how a fresh merge proposal
may be offered against the moved-on basis. This page does not restate those
rules; it relies on them.

### Proposals are permission-checked twice

Permission is evaluated at both ends of a proposal's life:

```text
at creation -> may this actor propose against this target?
at accept   -> may this actor apply this change?
```

The two checks are independent and both are required. A commenter may hold
propose rights without apply rights; an editor accepting a suggestion must
themselves be authorized to make the change, regardless of who proposed it.
The accept-time check happens at commit time inside the transaction, per
[Permissions](permissions.md).

This is also why proposals are safe as a cross-boundary channel: proposing
requires only the right to be heard, never the right to mutate.

### Surfacing is in place and in an inbox

Proposals appear in two ways:

- in place — a star shows its own open proposals, so suggested change is
  visible exactly where it would land;
- in a workspace inbox — all open proposals across the workspace, ordered by
  age, so nothing waits invisibly.

Both surfaces show the same records. Neither is a kernel structure; they are
views over proposal facts.

### Resolved proposals are sedimentary

Accepted, rejected, and superseded proposals are kept in history, out of the
way but inspectable. A rejection is a recorded judgment, not an erasure; a
superseded proposal remains as evidence of what was once offered against an
older basis. Redaction and tombstoning of proposal content follow the same
governance as any other record, per [Deletion](deletion.md).

## Working model

### Watched transclusion as a source-update proposal

The watched-transclusion flow settled in [Deep fates](deep-fates.md) is a
proposal flow expressed in this vocabulary. The occurrence renders a pinned
revision (`mode: "transclude"`, `pin: RevisionId`, `watch: true`). When the
source chunk gains a new current revision:

```text
source change
  -> source-update proposal
       basisRevisionIds: [the pinned source revision]
       targetChunkIds: [the containing document]
       payload: repin the occurrence to the new revision
  -> accept: occurrence repins to the new revision
  -> reject: pin unchanged
  -> either way: watch remains true; watching continues
```

`targetChunkIds` names the containing document because the type holds chunks;
the payload names the occurrence to repin. Freshness for an occurrence-targeted
proposal is checked against the occurrence, not the container's content: the
proposal is fresh while the occurrence's current pin still equals the basis
revision, and the containing document's own revisions do not stale it. If the
author repinned in the meantime, the accept is refused and the proposal is
superseded.

The local document never changes under the author's feet, and rejecting one
update does not end the relationship — the next source change produces the
next proposal.

### Generation as a proposal

`generate` records its `inputRevisionIds` — the reduced context — and yields a
`generation` proposal rather than content. Accepting it is what creates the
revision (or new chunk plus derivation): the revision records the model actor
as `createdBy` and the accept as its `operationId`, so provenance queries see
model material directly, while the admitting decision remains the accepting
actor's operation. The distinction
between "a model suggested this" and "an author admitted this into the work"
is kept as two facts, not collapsed into one.

### Detected relations as proposals

Soft similarity — echo-index matches, candidate referents from the
[Resolver](resolver.md) — remains derived index material until promoted. The
promotion offer is a `detected-relation` proposal; accepting it creates the
durable link. Indexes never write truth directly.

## Open questions

- Can a multi-part payload be partially accepted, or is a proposal always
  all-or-nothing, with partial acceptance modeled as a fresh narrower
  proposal?
- Should the inbox support prioritization beyond age — by kind, by target, by
  proposer — and is that ordering a personal setting or a view concern only?
- When supersession produces a replacement proposal, should the replacement
  link back to what it superseded as a durable relation or only as a query
  result?
