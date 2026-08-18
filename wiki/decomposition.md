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
article       -> one distant star
sections      -> ordered constellation when opened
sentences     -> derived stars when a section is focused
words         -> derived stars at closer inspection
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
interface DerivedSpan {
  sourceRevisionId: RevisionId;
  startOffset: number;
  endOffset: number;
  decompositionMethod: string;
}
```

This address can support inspection, temporary selection, search, and candidate
links. It becomes durable kernel identity only when promoted.

### Promotion records why it happened

```ts
interface Promotion {
  derivedSpan: DerivedSpan;
  newChunkId: ChunkId;
  operationId: OperationId;
  reason:
    | "edit"
    | "copy"
    | "fork"
    | "quote"
    | "transclude"
    | "annotate"
    | "move"
    | "link"
    | "permission";
}
```

The reason matters because different operations produce different structures.

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

### Copied chunk

The promoted part creates a new independent chunk derived from the source span.
The parent remains simple content.

Use when the user copies or forks material into a new context.

### Addressable span

The span receives durable addressability for annotation or reference, but the
parent does not necessarily become a composite structure.

Use when preserving the exact source location matters more than making a new
editable object.

## Product rule

The user should not feel the system creating thousands of objects while they
read, select, or inspect. Durable structure appears when the user commits to an
action that needs durable structure.

