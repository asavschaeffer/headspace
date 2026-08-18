# Operations

## Purpose

Operations translate user intent into reads, external work, and atomic kernel
changes. They compose primitives and seams; they are not all primitives
themselves.

Every kernel change is an operation, every operation is one atomic transaction,
and every transaction is one commit in the [Store](store.md). This page fixes
the transaction vocabulary, the select–reduce–generate pipeline, and the
proposal lifecycle. The semantics and taxonomy of proposals themselves live in
[Proposals](proposals.md).

## Accepted principles

### The transaction vocabulary

The kernel accepts a closed set of operation kinds. Each is one atomic commit:
a multi-object change such as an extraction is still a single transaction.

```ts
type OperationKind =
  | "create" | "revise"                       // content
  | "place" | "move" | "sever"                // occurrences
  | "relate" | "unrelate"                     // links
  | "copy" | "reference" | "transclude"       // reuse
  | "promote"                                 // span -> durable object
  | "propose" | "accept" | "reject"           // proposal lifecycle
  | "tombstone" | "redact";                   // governance

interface Operation {
  id: OperationId; kind: OperationKind; actorId: ActorId; at: string;
  inputRevisionIds: RevisionId[]; outputRevisionIds: RevisionId[]; params: unknown;
}
```

`place` adds an occurrence to a container, `move` repositions one, and `sever`
removes one. `relate` and `unrelate` manage links. `promote` turns a derived
span into a durable object in one of the settled shapes. `tombstone` and
`redact` are governed by [Deletion and redaction](deletion.md).

What each kind produces:

```text
revisions                  create, revise, accept (content payloads),
                           promote (extraction shape)
relations only             place, move, sever, relate, unrelate,
                           reference, transclude
new identity + derivation  copy, promote (copy shape), accepted generation
proposals only             propose
governance facts           tombstone, redact
```

Every operation is a transaction: one operation, one commit, one log append.

### View state is never a kernel commit

Selection, nebula arrangement, and lens choice are view material. They do not
pass through the operation layer and never appear in the commit log. Only
changes to truth — content, structure, relations, proposals, governance —
are operations. See [Views](views.md).

### Direct edits and proposals

Human edits in one's own workspace apply directly as `revise`. Everything that
arrives from elsewhere is a proposal first:

```text
human edit, own workspace      -> revise (direct commit)
model output                   -> proposal
watched-source update          -> proposal
collaborator suggested edit    -> proposal
detected relation              -> proposal
merge / reconciliation         -> proposal
```

Generation does not silently overwrite human work; neither does a driver, a
collaborator, or an index. A proposal is inert: its payload describes
operations, and nothing about the target changes until an accept applies them.

One scoped exception: when a bound file changed externally and Substrate holds
no internal edits since the last sync, the driver applies matched changes
directly as `revise` by `driver:fs` — an external-only edit is the author's
own edit arriving through the filesystem ([Drivers](drivers.md)). Divergence
on both sides is always a reconciliation proposal.

### Authorization happens at commit time

Permission checks run inside the transaction, at the moment of commit, through
the [Permissions](permissions.md) seam. An operation that fails its check does
not commit and leaves no partial state. Proposals are additionally checked at
creation — may this actor propose against this target? — as described in
[Proposals](proposals.md).

## The select–reduce–generate pipeline

Select, reduce, and generate remain the canonical pipeline for producing new
material from existing material. They are a pipeline over the transaction
vocabulary, not kernel primitives themselves: selection and reduction are
reads, and generation terminates in `propose`.

### Select

Choose candidate chunks and revisions for a purpose. Selection policy may use
focus, explicit user choices, hard relations, search, similarity, permissions,
and recency.

Selection should remain inspectable: the user should be able to understand
what was selected and why.

### Reduce

Compile a selection into a bounded representation. Reduction may order,
truncate, summarize, quote, or otherwise transform context while retaining
attribution to its sources.

A reduced context is structured, not a bare string, so that provenance is not
erased:

```ts
interface ReducedContext {
  items: { revisionId: RevisionId; chunkId: ChunkId; text: string; role: string }[];
  budget: number;
}
```

### Generate

Ask a human, model, or other transformer to produce output from reduced
context and an instruction.

Generation is a seam because providers and even the nature of the generator
are replaceable. Its result always returns as a Proposal, never an immediate
overwrite, and the recording operation carries `inputRevisionIds` equal to the
reduced set — the exact revisions the generator saw are part of
[Provenance](provenance.md).

## Proposal lifecycle

A proposal records what it was computed against and moves through a fixed
lifecycle:

```ts
interface Proposal {
  id: ProposalId;
  kind: "generation" | "source-update" | "suggested-edit"
      | "detected-relation" | "merge" | "reconciliation";
  status: "open" | "accepted" | "rejected" | "superseded";
  basisRevisionIds: RevisionId[];   // what the proposal was computed against
  targetChunkIds: ChunkId[];
  payload: ProposedChange[];        // inert operation descriptions, applied only on accept
  createdBy: ActorId; createdAt: string;
  resolution?: { by: ActorId; at: string; operationId?: OperationId };
}
```

```text
open -> accepted | rejected | superseded
```

- **accept** validates basis freshness: the state the basis names must still
  be current. For a content proposal, the target chunk's current revision must
  still be among `basisRevisionIds`; for an occurrence-targeted proposal such
  as a source-update, the occurrence's current pin must still equal the basis
  revision — the container's own content revisions do not stale it. If the
  check passes, the payload applies atomically as one commit and the
  resolution records the resulting operationId. If the target has moved on,
  the accept is refused and the proposal flips to superseded; the system may
  offer a fresh merge proposal computed against the new basis
  (see [Conflicts](conflicts.md)).
- **reject** is sedimentary: the proposal and its resolution are kept in
  history, not erased.
- **superseded** happens automatically when the basis goes stale or when a
  newer proposal for the same target and kind replaces this one.

Freshness is the invariant that makes proposals safe: a payload only ever
applies against the state it was computed from.

## Intentions built from operations

```text
edit          -> revise
move          -> move (reposition an occurrence), or sever + place
contextualize -> select (+ optionally reduce); no kernel commit
reroll        -> select + reduce + generate -> propose
accept        -> accept (applies the proposal payload atomically)
branch/fork   -> copy (new identity + derivation)
delete        -> sever, tombstone, or redact (see Deletion)
```

## Reuse operations

Substrate has three core reuse verbs:

- copy: create a new identity from a source revision;
- reference: point to another identity or revision;
- transclude: render another identity or revision inline.

For authored documents, watched transclusion is the recommended default:

```text
render a pinned revision, watch the source for changes, ask before updating
```

This supports living source relationships without silently changing local
authorship.

If transclusion requires consent, a requested transclusion should initially
display as a reference. Approval may upgrade it to watched or live
transclusion. Denial leaves it as a reference or lets the requester remove it.

In the vocabulary above, the watched flow is: a source change produces a
`source-update` proposal targeting the containing document; its payload names
the occurrence to repin. Accept repins the occurrence to the new revision;
reject keeps the existing pin; watching continues either way. See
[Deep fates](deep-fates.md) for the relationship model.

## Open questions

- The `params` schema for each operation kind is unspecified beyond "enough to
  replay intent"; per-kind schemas are still to be fixed.
- Whether reduced contexts should be cached or recorded as durable facts, or
  remain ephemeral inputs whose provenance lives only in the generation
  operation's `inputRevisionIds`.
- Whether a single user gesture that implies several independent operations
  (a multi-item drag, a bulk sever) may ever fold into one commit, or must
  remain one commit per operation.
- Publication and unpublish ([Collaboration](collaboration.md)) are explicit
  operations whose kinds are not yet in the closed vocabulary; they are added
  when shared corpora exist.
