# Permissions

## Purpose

Permissions decide who may see or perform actions on chunks, revisions,
occurrences, relations, and deep-fate records.

Permissions are separate from provenance. Provenance records what happened;
permissions decide what may happen next and what may be revealed.

## Accepted principles

### Protective ground rules come first

Some rules should be system-level safety invariants, not preferences.

Examples:

- transclusion does not grant edit authority over the source;
- live transclusion requires explicit permission;
- denied transclusion does not silently become a copy;
- private source content must not leak through indexes, counts, or fate graphs;
- redacted speech must not remain publicly attributable to the redacting user;
- destructive or public propagation requires explicit authority.

Personal settings may customize behavior inside those boundaries, but they
cannot override protective ground rules.

### Personal settings tune defaults

Users may configure mid-level defaults for their own workspace and authored
documents.

Examples:

- prefer pinned references, watched transclusions, or live transclusions when
  allowed;
- show source changes as quiet notifications or inline diffs;
- auto-accept updates from trusted collaborators for low-risk documents;
- require review for public, published, or archival documents;
- decide when soft deep-fate suggestions appear.

These are policy preferences, not kernel invariants.

### Private content stays out of global indexes by default

Private workspace indexes should be separate from shared or public indexes.

Default model:

```text
private content -> private index only
shared/public corpus -> explicit opt-in
```

This protects against count leaks, first-seen leaks, author inference, and
accidental discovery through deep-fate queries.

Permission-filtered global indexes may be possible later, but they are not the
recommended foundation because subtle filtering mistakes can expose private
information.

### Permissions inherit downward by default

Permissions attach high in the containment tree by default and inherit
downward.

```text
workspace
  project
    document
      chunk
        occurrence
          derived span/token
```

Overrides should be sparse and explicit. Derived parts inherit from their
nearest durable parent unless promoted into durable objects.

Principle:

```text
Permission granularity follows promotion granularity.
```

This keeps both system memory and human memory manageable.

### Container policy and source policy both apply

Permission inheritance follows two axes:

1. local containment;
2. source/reuse relationship.

The local container controls local visibility. The source controls reuse
obligations.

An operation must satisfy both.

Example:

```text
Asa's private document contains Mira's watched transclusion.

Local container policy:
  Asa's document is private.

Source policy:
  Mira permits watched transclusion but requires attribution.

Result:
  the occurrence is visible only inside Asa's private document;
  Mira's attribution and update policy are preserved;
  neither party gains edit authority over the other's object.
```

This protects both sides of a reuse relationship.

### Reuse defaults

Substrate's default reuse model is:

```text
visible material may be referenced
copying depends on source policy
watched/live transclusion requires explicit source permission
editing source requires edit permission
```

Reference is the weakest relationship. It points to another object without
embedding it as local content or creating an ongoing dependency.

Transclusion is stronger because it embeds another object's wording into the
local work and may create an ongoing relationship to future revisions.

### Deep-fate visibility is policy-based

The visibility of downstream reuse should depend on the reuse verb and source
policy.

Default model:

```text
private references/copies -> private unless shared or published
transclusion requests     -> visible to the source owner because consent is needed
published reuse           -> creates visible deep-fate links
source policy             -> may require attribution, notification, or approval
```

These are permission and visibility decisions. The kernel should record enough
truth to support them, but it should not hard-code one social policy for every
workspace.

User and workspace settings may choose stricter or looser defaults inside the
protective ground rules.

### Pending transclusion displays as reference

If a user requests transclusion and the source requires consent, the local
document should display the relationship as a reference while waiting.

```text
requested transclusion
  -> pending permission
  -> render as reference
  -> source owner approves or denies
```

If approved, the reference may become watched or live transclusion according to
the requested mode and granted permission.

If denied, the relationship remains a reference or is removed by the requesting
user.

### Watched transclusion update flow

Watched transclusion renders a pinned source revision and watches the source
for changes.

When the source changes, the default behavior is:

```text
source revision changes
  -> local occurrence remains pinned
  -> Substrate shows an inline diff/proposal
  -> local author accepts, rejects, or keeps watching
```

The source owner may request propagation to downstream watchers, but should not
be able to force an update into another author's document unless a stronger
permission relationship explicitly allows it.

### Edit authority is explicit

Copying, referencing, deriving from, or transcluding another object does not
grant edit authority over the source.

Editing the source requires explicit edit permission.

## Relationship to other seams

- Janus protects the boundary between reuse types.
- Provenance records who requested, approved, denied, or performed an action.
- Deep Fates reveal relationships only when visibility policy allows them.
- Operations must check permissions before committing kernel changes.
