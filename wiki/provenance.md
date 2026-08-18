# Provenance

## Purpose

Provenance records where material came from, who or what changed it, when that
happened, and by which operation.

It is sedimentary: normally out of the user's way, but available for debugging,
trust, attribution, authorship, recovery, analytics, and deep fates.

## Accepted principles

### Default authorship is operation-level

Substrate's default authorship model is operation-level authorship.

The current revision has a creator, and every transformation records the
operation that produced it. Span-level authorship can be derived when needed,
but every surviving word does not become socially owned by default.

```ts
interface Revision {
  id: RevisionId;
  chunkId: ChunkId;
  blobHash: BlobHash;
  mediaType: string;
  parentRevisionIds: RevisionId[];
  createdBy: ActorId;
  createdAt: string;
  operationId: OperationId;
}

interface Operation {
  id: OperationId;
  actorId: ActorId;
  kind:
    | "create" | "revise"
    | "place" | "move" | "sever"
    | "relate" | "unrelate"
    | "copy" | "reference" | "transclude"
    | "promote"
    | "propose" | "accept" | "reject"
    | "tombstone" | "redact"
    | "import" | "reconcile";
  inputRevisionIds: RevisionId[];
  outputRevisionIds: RevisionId[];
  patch?: Patch;
}
```

Generation is not an operation kind of its own: model output reaches truth
through `propose` and `accept`, so who suggested and who admitted are recorded
as separate facts.

Example:

```text
source: "the lake is still"   by original author
edit:   "the lake is stormy"  by Asa
```

The current revision is Asa's authored revision. Provenance records that it was
derived from the original and that the span `the lake is` survived the edit.

### Provenance is not permission

Provenance records what happened. Permissions decide what may happen next and
who may see what happened.

Do not collapse source history into edit authority.

### Provenance is not social attribution by itself

The system may know that a phrase, token, or span appeared elsewhere first.
That does not mean the interface must display ownership for every word.

Social attribution is governed by visibility policy, source policy, and user
intent.

## Relationship to other seams

- Janus decides boundary questions such as same object, copy, fork, derivation,
  and coincidence.
- Deep Fates use provenance to explain where material came from and where it
  went.
- Permissions decide who may view, copy, reference, transclude, edit, publish,
  or delete.
- Lenses may derive span-level authorship or style analytics from provenance.

