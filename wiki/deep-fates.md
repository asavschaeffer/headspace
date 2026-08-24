# Deep fates

## Purpose

Deep fates are the visible histories and relationships that connect repeated,
copied, referenced, transcluded, and transformed material across the workspace
graph.

The user-facing question is:

> Where has this thought, phrase, image, fragment, or structure gone?

The kernel-facing question is:

> Which identities, revisions, blobs, occurrences, and relations connect this
> object to other objects?

## Core idea

When a writer creates `the lake is still`, and another writer later creates the
same phrase, the workspace graph may record a relationship between them without
making them the same editable object.

The relationship can say:

- this content is byte-identical;
- this phrase appears to derive from an earlier phrase;
- this object copied that revision;
- this object references that source;
- this document transcludes that phrase;
- this later phrase transformed part of the earlier phrase.

Deep fates are not only history. They are the living graph of where material
has appeared, forked, echoed, and changed.

Deep Fates connect more than blobs. They may connect blobs, spans, revisions,
chunks, occurrences, authors, operations, permissions, and visible contexts.
Blob equality can say that content is identical; Deep Fates explain what that
identity means in context.

## Division of responsibilities

### Janus guards thresholds

Janus answers boundary questions:

- Are these two things the same object?
- Are they separate objects with identical content?
- Is this a copy, reference, transclusion, derivation, or coincidence?
- Did this edit preserve identity or create a fork?

Janus protects the system from collapsing identity, content equality,
provenance, and permission into one ambiguous claim.

### Deep Fates tie material together

Deep Fates answer relationship questions:

- Where did this come from?
- Where has it gone?
- Who reused it?
- What changed?
- Which spans survived transformation?
- Which descendants still point back?

Deep Fates are the graph of living relationships between material.

### Provenance and permissions govern authority

Provenance records who or what made something, when, from which source, and by
which operation.

Permissions decide who may view, copy, reference, transclude, edit, publish, or
delete something.

Copy, reference, and transclusion are therefore policy-bearing operations, not
just UI gestures.

Default reuse policy:

```text
visible material may be referenced
copying depends on source policy
watched/live transclusion requires explicit source permission
editing source requires edit permission
```

If transclusion permission is pending, the requesting document should display
the relationship as a reference until the source owner approves or denies.

## The three reuse verbs

### Copy

Copy creates a new identity from an existing revision.

```text
new chunk --copied-from--> source revision
```

The new chunk starts with the same content but edits independently. Copy keeps
provenance without granting mutation rights over the original.

### Reference

Reference points at another chunk or revision without embedding it as editable
local content.

```text
local chunk --references--> source chunk/revision
```

Reference is useful for citation, callback, inspiration, dependency, or
association.

### Transclude

Transclusion renders another chunk or revision inline.

```text
document occurrence --transcludes--> source chunk/revision
```

Transclusion may be live or pinned:

- live transclusion follows the source chunk's current revision;
- pinned transclusion shows a specific revision forever unless changed.

Transclusion does not automatically imply permission to edit the source.

### Watched transclusion

Watched transclusion is the recommended default for reuse inside authored work.

```text
document occurrence --watches--> source chunk
document occurrence --renders--> pinned source revision
```

The user's document remains stable because it renders a pinned revision. If the
source changes, the client can notify the user and offer an explicit update.

This preserves authorship while keeping the deep fate alive.

Use cases:

- quoting someone whose word choice matters;
- embedding a collaborator's sentence while retaining review control;
- tracking an evolving source without letting it silently rewrite local work.

When the source changes, watched transclusion should behave like an update
proposal: the local document remains pinned, shows the difference, and lets the
local author accept or reject the change.

## Visibility from both directions

Deep fates should be queryable forward and backward.

From the child:

```text
Where did this come from?
```

From the parent:

```text
Where did this go?
Who copied it?
Who referenced it?
Who transcluded it?
What did it become?
```

This allows an original author to see that another user made a child phrase
elsewhere, if permissions and visibility rules allow it.

Default visibility is policy-based:

```text
private copies/references may stay private
transclusion requests are visible because consent is needed
published reuse creates visible deep-fate links
source policy can require attribution, notification, or approval
```

## Identity and permission

Deep fate does not mean shared mutability.

```text
same content      -> may share blob storage
same ancestry     -> may be fate-linked
same chunk id     -> same editable object
edit permission   -> explicit capability
```

Copying or referencing another object does not grant permission to mutate the
original. Live transclusion displays the original but still does not imply edit
permission unless separately granted.

Default rule:

```text
Authored documents should prefer watched transclusion over live transclusion.
```

Live transclusion is allowed, but should be explicit because it can make a
document change under the author's feet.

## Authorship

Deep Fates use operation-level authorship by default.

The current revision has a creator. The operations that produced it record
their inputs and sources. Span-level authorship can be derived when a user asks
to inspect deep provenance, but normal writing should not show every word as
socially owned by someone else.

## Edit behavior

If `the lake is still` becomes `the lake is stormy`, Headspace can preserve
partial fate:

```text
the lake is still
the lake is stormy
```

The phrase changed, but the span `the lake is` remains shared. The changed span
breaks exact equality while preserving ancestry and partial overlap.

This does not require every word to become a durable chunk. Most subphrase
matches can remain derived index facts until the user promotes a span.

## Data layers

Deep fates use several layers:

1. content-addressed storage for exact blob equality;
2. span indexes for repeated phrases, subphrases, and echoes;
3. kernel relations for explicit copy/reference/transclusion/derivation;
4. provenance records for authorship, time, source, and operation history;
5. permission rules for who may see or mutate each object.

## Product rule

The interface should keep deep fates sedimentary.

Most of the time, users write normally. When they ask where something came
from, where it went, or why it changed, the system can reveal the underlying
fate graph.

## Redaction and reflow

First-seen attribution is derived from visible, permitted, non-redacted
evidence. It is not a permanent crown.

If an earlier source redacts their instance of a phrase, downstream independent
instances remain intact. Public attribution may reflow to the next eligible
source, or the prior source may appear as a blank redacted placeholder depending
on policy.
