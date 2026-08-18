# Drivers

## Purpose

A driver translates an external representation into kernel objects or
projects kernel objects back into an external representation.

Examples include Markdown files, directories, LLM conversation exports,
web pages, and future application-specific formats.

A driver interprets; the kernel records. The kernel stores what a driver
produced using revisions, occurrences, derivations, and links, and never
learns the external format itself (see [Information shapes](information-shapes.md)).

## Seam

A driver is replaceable behind a capability-oriented contract. A read-only
importer and a bidirectional filesystem driver implement different
capabilities rather than pretending every source is writable.

```ts
interface ImportDriver {
  detect(source: ExternalSource): Promise<Confidence>;
  import(source: ExternalSource): Promise<ImportResult>;
}

interface ExportDriver {
  project(change: KernelChange, binding: Binding): Promise<ProjectResult>;
}

interface ReconcileDriver {
  reconcile(binding: Binding): Promise<ReconcileResult>;
}
```

These interfaces are sketches, not settled APIs. The Markdown driver
implements all four capabilities: detect, import, project, reconcile.

Chunk decomposition is not driver-private policy. A driver chooses *which*
decomposition applies to its format, but the segmentation itself is a
registered decomposer method — the Markdown driver uses `md/blocks@1` —
so addresses and derived parts stay comparable across drivers (see
[Decomposition](decomposition.md)).

## Responsibilities

- Detect and parse an external form.
- Choose a useful chunk decomposition from the method registry.
- Preserve enough source information for faithful projection.
- Preserve chunk identity across repeated imports of the same external
  object, via sidecar memory.
- Report unsupported or lossy translations explicitly; nothing is silently
  dropped.
- Attribute imported revisions to their driver actor (e.g. `driver:fs`).
- Surface divergence between the store and the external object; never
  resolve a two-sided conflict on its own authority.

## Non-responsibilities

- Owning chunk identity rules.
- Persisting universal history.
- Ranking search results.
- Choosing what an agent sees.
- Deciding conflicts: a driver detects divergence and files a
  reconciliation [proposal](proposals.md); acceptance is a user act.

## The Markdown driver

The first driver is Markdown over the filesystem, acting as `driver:fs`.
It exercises every capability of the seam and proves round-tripping,
identity preservation, and reconciliation against real files.

### Import

One file becomes one doc chunk (a composite) plus one block chunk per
`md/blocks@1` segment: headings, paragraphs, list blocks, code fences.
Block chunks appear in the doc through ordered occurrences.

The doc revision retains the exact source blob (mediaType
`text/markdown`) alongside the derived structure while both exist — the
third working-model case in [Content](content.md). Exact bytes preserve
faithful round-trip; structure makes the parts addressable.

Lossiness is declared, not hidden: opaque HTML blocks pass through as
unparsed blocks. They survive round-trip byte-for-byte; they are simply
not interpreted further yet.

### Sidecar

Round-trip memory lives in a sidecar owned by the driver, stored under
the workspace [store](store.md) — never as inline markers in the `.md`
file itself. The external file stays clean.

```text
.substrate/sidecars/<relpath>.json
```

```ts
interface MarkdownSidecar {
  docChunkId: ChunkId;
  bindingId: BindingId;
  blocks: { chunkId: ChunkId; blobHash: BlobHash }[];  // in occurrence order
  lastProjectedFileHash: BlobHash;
}
```

The sidecar is how identity survives repeated imports: re-reading a file
matches its blocks back to the chunks they already are, rather than
minting new identities each time. If a sidecar is lost, identity is
reconstructed by content-hash matching; unmatched blocks become new
chunks and the loss is reported, not hidden (see [Conflicts](conflicts.md)).

### Project (export)

Projection assembles block content in occurrence order, writes the file,
then updates the sidecar and the binding's `observedVersion`. The
[binding](bindings.md) targets the doc chunk; its observed version is the
external content hash. One chunk may carry multiple bindings as export
targets, while one file binds one doc chunk. A renamed file is
rediscovered by content-hash match and offered as a `reconciliation`
proposal to rebind.

### Reconcile

An external edit is detected when the file's hash differs from
`lastProjectedFileHash`. The driver then matches the file's current
blocks against the sidecar's chunks:

```text
1. exact match        -> same blobHash: block unchanged
2. positional match   -> sidecar order + similarity >= 0.5
                         (normalized edit distance): block edited
3. unmatched in file  -> new chunk
4. unmatched in sidecar -> sever of the occurrence, proposed within
                           the reconciliation proposal
```

Fast path: if Substrate holds no internal edits since the last sync,
matched-changed blocks apply directly as `revise` operations by actor
`driver:fs` — an external-only edit is just an edit, arriving through the
driver.

If both sides changed, the driver files a reconciliation proposal
carrying its computed matching; the divergence resolves through the merge
machinery in [Conflicts](conflicts.md). Never silent loss in either
direction: external edits cannot vanish under a projection, and Substrate
edits cannot vanish under an import.

Divergence handling follows the authority split: the store is
authoritative for identity, structure, history, and provenance; the bound
file is authoritative for bytes edited outside Substrate. The driver is
the border crossing where the two are brought back into agreement.

## Open questions

- Which driver comes next: a directory tree, or LLM conversation exports
  as mapped in [Information shapes](information-shapes.md)?
- Should the 0.5 similarity threshold be tunable per driver or per
  binding?
- How do sidecars travel when a workspace later syncs across machines?
- When do opaque HTML blocks gain real parsing rather than pass-through?
