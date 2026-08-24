# Decomposition and promotion

## Purpose

Decomposition exposes smaller parts of content on demand. Promotion turns a
derived part into durable kernel truth when the user commits to treating it as
an object.

## Accepted principles

### Granularity is progressive and demand-driven

The same material may appear at different levels of resolution without eagerly
minting a durable chunk at every level.

```text
article        -> one distant star
sections       -> ordered constellation when opened
sentences      -> derived stars when a section is focused
words          -> derived stars at closer inspection
subword tokens -> tokenizer-specific derived stars when explicitly needed
```

A decomposer exposes the next useful layer from a specific immutable revision.
Views may render those derived addresses as stars and cache their structure.
Merely zooming, focusing, or inspecting does not promote them into kernel
identity.

Subword tokens are especially interpretation-dependent: different tokenizers
produce different boundaries. Their addresses therefore record the method and
version that produced them rather than pretending to be universal content
truth.

### A decomposer is a pure function over one immutable revision

Decomposition never mutates the kernel. A decomposer takes a revision and a
method and returns derived parts:

```ts
interface DerivedPart {
  address: SpanAddress; // { revisionId, method, start, end }
  kind: "block" | "sentence" | "word";
  text: string;
}

decompose(revision: Revision, method: string): DerivedPart[];
```

Because the input is an immutable revision and the method string is versioned,
the same call always returns the same parts. Offsets are UTF-16 code-unit
offsets into the decoded text of the revision's blob, so an address remains
meaningful for as long as the revision exists — which, in a sedimentary store,
is indefinitely.

### Methods are registered and versioned

A method string names both the segmentation strategy and its version, because
a boundary is a fact about an interpreter, not about the text alone.

```text
md/blocks@1  -> markdown block segmentation (headings, paragraphs,
                list blocks, code fences)
sent/icu@1   -> sentence segmentation via Intl.Segmenter
word/icu@1   -> word segmentation via Intl.Segmenter
```

Subword tokenizers are deferred; the address format already records
method+version, so adding a tokenizer later changes the registry, not the
model.

### Decomposition results are cached, and the cache is disposable

Results are cached by `(revisionId, method)` in the [Index](index.md) layer.
Since revisions are immutable and methods are versioned, entries never go
stale — they are simply evicted or rebuilt at will. Losing the cache loses no
truth.

### Attention does not promote; commitment promotes

Derived parts should not become durable chunks merely because the user looks at
them.

```text
looking at a span      -> no promotion
selecting/highlighting -> no promotion by itself
inspecting provenance  -> no promotion
opening as nebula      -> no promotion by itself
```

Promotion happens when the user asks the system to preserve, transform, relate,
govern, or reuse a derived part as an object.

```text
copying                -> promotion likely
forking                -> promotion
quoting/transcluding   -> promotion
annotating durably     -> promotion or durable span address
editing directly       -> promotion
moving/reordering      -> promotion
creating explicit link -> promotion
applying permissions   -> promotion
```

### Storage, editing, and interaction granularity are separate

A paragraph may be stored as one content blob, edited as a paragraph, inspected
as sentences, and indexed as tokens.

Those granularities should not be forced to match.

```text
storage unit     != editing unit
editing unit     != interaction unit
interaction unit != index unit
```

### Derived spans are addressable before they are chunks

A decomposer can expose a span without creating a durable chunk.

```ts
interface SpanAddress {
  revisionId: RevisionId;
  method: string; // registered, versioned decomposition method
  start: number;  // UTF-16 code-unit offset
  end: number;
}
```

This address can support inspection, temporary selection, search, and candidate
links. It becomes durable kernel identity only when promoted.

### Promotion is one atomic operation, and it records why it happened

Promotion is a single kernel transaction:

```ts
promote(span: SpanAddress, reason: PromotionReason, shape: PromotionShape);

type PromotionReason =
  | "edit"
  | "copy"
  | "fork"
  | "quote"
  | "transclude"
  | "annotate"
  | "move"
  | "link"
  | "permission";

type PromotionShape = "extract" | "copy" | "addressable-span";
```

Everything a promotion produces — new chunk, new parent revision, occurrences,
derivation, links — lands in one commit, so no observer ever sees a
half-promoted state. The reason matters because different operations produce
different structures, and the [Operation](operations.md) record preserves the
intent alongside the resulting facts.

## Promotion shapes

Promotion does not always mean extraction.

```text
quote/transclude -> reference target or transclusion source
fork/copy        -> copied chunk
edit/move        -> extraction
annotation       -> addressable span, not necessarily extracted
```

### Extracted chunk

The promoted part becomes a child chunk, and the parent structure changes to
contain or transclude it.

Use when the user edits, moves, or rearranges the part as part of the original
object.

When extraction makes a parent composite, the parent gains a new internal
composite revision whose content lists
its child occurrence ids in order. The children carry the parts; the
unextracted remainder becomes the minimal set of sibling chunks — the
contiguous spans before and after the extracted part — so the composite covers
the whole text without minting one chunk per derived part of the decomposition
method. An [adapter](adapters.md) reassembles Markdown
from the composite on export. The prior flat revision remains in history — the
parent's identity continues, and its earlier byte-for-byte form stays
recoverable (the layered representation described in [Content](content.md)).

### Copied chunk

The promoted part creates a new independent chunk with a
[Derivation](kernel.md) via `extract` or `copy` recording the source revision
and span. The parent is untouched — no new parent revision, no composite.

Use when the user copies or forks material into a new context.

### Addressable span

The span's `SpanAddress` is registered durably as a link or annotation anchor.
No new chunk is created, and the parent does not become a composite structure.

Use when preserving the exact source location matters more than making a new
editable object.

## Product rule

The user should not feel the system creating thousands of objects while they
read, select, or inspect. Durable structure appears when the user commits to an
action that needs durable structure.
