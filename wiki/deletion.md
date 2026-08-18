# Deletion and redaction

## Purpose

Deletion governs how Substrate removes, hides, tombstones, severs, or redacts
objects without corrupting identity, provenance, permissions, indexes, or deep
fates.

## Accepted principles

### Deletion has multiple meanings

There is not one deletion operation.

```text
sever occurrence -> remove this appearance from this container
tombstone        -> mark an object deleted while preserving identity/history
unpublish        -> remove shared/public visibility
redact           -> hide content and/or attribution
hard delete      -> destroy stored content/history where policy allows
```

Ordinary user deletion should usually sever an occurrence or tombstone a local
object. Published or shared material usually needs unpublish or tombstone.
Privacy/legal emergencies may need redaction or hard delete.

**Product default:** deleting a star severs the occurrence the user is looking
at; tombstoning the identity everywhere is an explicit, distinct action.

### The right to destroy one's own speech exists

A user should have a path to remove what they said so that the system no
longer presents it as something they said.

This is stronger than hiding an object from a view. It affects attribution,
deep fates, indexes, and public provenance.

### Redaction must not corrupt later independent speech

If the first known author of a phrase redacts their instance, later independent
instances should remain intact.

Example:

```text
Mira: "the lake is still"   first seen
Asa:  "the lake is still"   second seen
Mira redacts her instance
```

After redaction, public attribution should not keep claiming Mira said it.

Possible public outcomes:

```text
prior source: [redacted]
first visible source: Asa
```

or:

```text
first visible source: Asa
```

Which display policy to use is configurable. The settled architectural point is
that Mira is either redacted or replaced in visible attribution. Everything
else is downstream.

### First-seen is a query result, not a permanent crown

The "first" source for a phrase is derived from currently visible, permitted,
and non-redacted evidence.

If an earlier source is removed or redacted, first-seen attribution can reflow
to the next eligible source.

This prevents deep fates from permanently exposing someone who exercised a
right to remove their speech.

### Redaction can leave a blank

Depending on policy, a redacted source may appear as a blank placeholder rather
than disappear entirely.

```text
source: [redacted]
relationship: existed but content/author hidden
```

This is useful when downstream graph integrity matters but the redacted user
should no longer be named or quoted.

## Engineering constraint

Support the full vocabulary without making every deletion path expensive.

The practical implementation can start with:

1. sever occurrence;
2. tombstone object;
3. unpublish;
4. redact content/attribution from visible indexes.

Hard delete can remain a rarer administrative or retention operation.

