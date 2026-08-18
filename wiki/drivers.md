# Drivers

## Purpose

A driver translates an external representation into kernel objects or
projects kernel objects back into an external representation.

Examples include Markdown files, directories, LLM conversation exports,
web pages, and future application-specific formats.

## Seam

A driver should be replaceable behind a capability-oriented contract. A
read-only importer and a bidirectional filesystem driver may implement
different capabilities rather than pretending every source is writable.

Possible capabilities include:

```ts
interface ImportDriver {
  detect(source: ExternalSource): Promise<Confidence>;
  import(source: ExternalSource): Promise<ImportResult>;
}

interface ExportDriver {
  project(change: KernelChange, binding: Binding): Promise<ProjectResult>;
}
```

These interfaces are sketches, not settled APIs.

## Responsibilities

- Detect and parse an external form.
- Choose a useful chunk decomposition.
- Preserve enough source information for faithful projection.
- Report unsupported or lossy translations.
- Attribute imported revisions to their source.

## Non-responsibilities

- Owning chunk identity rules.
- Persisting universal history.
- Ranking search results.
- Choosing what an agent sees.

## Open questions

- Is chunk decomposition driver policy or a separate parser seam?
- How are IDs preserved across repeated imports?
- What happens when an external edit and a Substrate edit conflict?
- How does a driver declare lossless versus lossy write-back?
- Which first driver best proves the architecture: Markdown or a directory
  tree?

