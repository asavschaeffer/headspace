# Collaboration

## Purpose

Collaboration defines who acts in Substrate, what authority each actor holds,
and how work moves between actors and workspaces without anyone's document
being changed behind their back.

Substrate is collaborative from the first day, even with one human in it. A
workspace already contains several kinds of actors — a human, model agents,
filesystem drivers, external layers — and every kernel operation already
records which one acted. Multi-human collaboration extends this model; it does
not introduce a new one.

## Decisions

### Every actor is a first-class identity

```ts
interface Actor {
  id: ActorId;      // "human:asa", "agent:<model>", "driver:fs", "external:<layer>"
  kind: "human" | "agent" | "driver" | "external";
  name: string;
}
```

Every operation records an `actorId` ([Provenance](provenance.md)). There are
no anonymous kernel changes. A model that generates text, a driver that
reconciles a file, and a human who revises a paragraph are the same kind of
fact at the kernel level: an actor performed an operation.

The single-human workspace is the first collaboration case, not a special
case. One human, several agents, and drivers already exercise the full
machinery: direct edits for the trusted actor, proposals for everyone else,
authorship queries by actor kind, provenance lenses that color human against
model material.

### Roles gate operations

Workspace membership assigns each actor a role:

```ts
interface Membership {
  actorId: ActorId;
  role: "owner" | "editor" | "commenter" | "viewer";
}
```

Roles are cumulative:

```text
viewer     -> read what visibility policy allows
commenter  -> viewer    + propose (suggested edits, relations, transclusion requests)
editor     -> commenter + accept/reject proposals, revise, place, relate
owner      -> editor    + membership, publication, tombstone, redact, policy
```

Role checks are enforced in the operation layer from the start, at commit time
inside the transaction ([Operations](operations.md)) — even while one human
holds every human role. The gates exist before the second person arrives, so
adding a collaborator is a membership fact, not an architectural event.

Groups are actor sets: a membership or a source policy may name a group where
it would name an actor. Group semantics stay above the kernel; the kernel
records which concrete actor performed each operation.

### Cross-workspace flows ride the proposal system

Collaboration needs no second channel. Everything that crosses an authority
boundary is a [Proposal](proposals.md) or a pending permission:

```text
suggested edit from a commenter   -> "suggested-edit" proposal
transclusion consent request      -> pending permission, rendered as reference
watched source changed upstream   -> "source-update" proposal
two editors diverged              -> "merge" proposal
```

A transclusion consent request is the one flow carried by the
[Permissions](permissions.md) seam rather than a Proposal record: it renders
as a reference until the source owner approves (see
[Deep fates](deep-fates.md)).

A commenter can propose; an editor can accept. The proposal carries its basis
revisions, so a stale suggestion is refused and superseded rather than applied
against text that has moved on. This is the same lifecycle that governs model
output and driver reconciliation — collaborators are actors with lower default
trust, not a different mechanism.

Notifications are the proposal inbox. There is no separate notification
system to keep consistent: an open proposal targeting your material is the
notification, sedimentary once resolved.

### Publication is an explicit operation

Material enters a shared or public corpus only through an explicit publication
operation by an actor with authority to publish ([Permissions](permissions.md)).
Nothing becomes visible outside its workspace as a side effect of editing,
indexing, or reuse. Publish and unpublish are governance operation kinds not
yet in the closed vocabulary of [Operations](operations.md); they are added
there when shared corpora land.

Unpublish, tombstone, and redaction follow [Deletion](deletion.md): published
material can be withdrawn, attribution can reflow, and redacted sources render
as placeholders downstream rather than corrupting other people's documents.

### Sync is deferred; the seam is the commit DAG

Multi-device and multi-user synchronization is deliberately not built yet. The
seam it will plug into already exists: commits form a DAG with plural parents
([Store](store.md)), so two replicas that advanced independently reconcile by
committing a merge with both histories as parents.

There is no CRDT commitment. Divergence resolves through merge proposals
([Conflicts](conflicts.md)) — clean combinations may auto-apply under a
personal setting, overlapping changes surface both sides verbatim for a human
decision. A future sync layer transports commits; it does not get to invent
new resolution semantics.

## Working model

```text
Asa's workspace, one human:

  human:asa        owner      revises directly; accepts and rejects
  agent:<model>    commenter  generation always lands as proposals
  driver:fs        editor*    fast-path revise on clean external edits;
                              proposals when both sides changed
  external:<layer> viewer     read-only cached snapshots, never chunk content
                              without an explicit copy

* the driver's edit authority is scoped to bound chunks and to the
  reconciliation rules in [Drivers](drivers.md); it is not a general editor.

Later, Mira joins as commenter:

  Mira reads what visibility policy allows.
  Her suggested edit is a proposal in Asa's inbox.
  Her transclusion request renders as a reference until Asa consents.
  Nothing Mira does mutates Asa's chunks until an editor accepts.
```

## Relationship to other seams

- [Permissions](permissions.md) decides what a role may see and do; container
  policy and source policy both still apply to every cross-actor operation.
- [Operations](operations.md) enforces role checks at commit time and owns the
  proposal lifecycle.
- [Proposals](proposals.md) is the vehicle for every cross-authority change.
- [Conflicts](conflicts.md) resolves divergence between actors and replicas.
- [Provenance](provenance.md) records which actor did what; [Deep Fates](deep-fates.md)
  reveal cross-actor reuse only where visibility policy allows.
- [Deletion](deletion.md) governs unpublish and the right to withdraw one's
  speech from shared corpora.

## Open questions

- How are `ActorId`s authenticated and kept stable across workspaces and
  machines once a second human or a remote replica exists?
- What transport carries commits between replicas, and how is partial sync
  (one shared project inside a private workspace) scoped?
- Do groups need their own membership history and roles, or do they remain
  simple actor sets resolved at check time?
