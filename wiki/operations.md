# Operations

## Purpose

Operations translate user intent into reads, external work, and atomic kernel
changes. They compose primitives and seams; they are not all primitives
themselves.

## The three broad operations

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

A reduced context should probably be structured rather than only a string so
that provenance is not erased.

### Generate

Ask a human, model, or other transformer to produce a proposal from reduced
context and an instruction.

Generation is a seam because providers and even the nature of the generator
are replaceable. Its result returns as chunk-shaped content with provenance,
usually as a proposal rather than an immediate overwrite.

## Intentions built from operations

```text
edit          -> revise
move          -> unrelate + relate
contextualize -> select (+ optionally reduce)
reroll        -> select + reduce + generate + propose
accept        -> revise or create + relate
branch        -> create identity + derivation relation
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

When a watched source changes, the update should enter the local document as a
proposal rather than a silent overwrite. The local author can accept, reject,
or keep watching.

## Open questions

- Are select, reduce, and generate the universal operations, or are they one
  important pipeline over more fundamental reads and writes?
- Is human editing a form of generate, or should “generate” be renamed to a
  more inclusive transformation concept?
- What is the formal lifecycle of a proposal?
- Which operations require transactions?
- How are authorization, validation, and conflicts surfaced?
