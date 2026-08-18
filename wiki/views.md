# Views

## Purpose

A view chooses how kernel objects and indexed relationships appear for a
particular task. The Nebula and Star are views, not kernel concepts.

## Accepted distinctions

### A visible star need not be a durable chunk

A Star is a view role. It may project a durable chunk or an addressable derived
part exposed on demand by a decomposer.

```ts
type ViewTarget =
  | { kind: "chunk"; chunkId: ChunkId }
  | { kind: "derived"; address: DerivedAddress };
```

Promotion is required only when an operation needs independent durable identity.
This allows progressive zoom without filling the kernel with every sentence,
word, and tokenizer-dependent subword.

Identity, semantic position, and visual placement are separate:

1. **Identity** answers which continuing thing this is.
2. **Semantic position** describes what a revision resembles.
3. **Visual placement** chooses where the chunk appears in a particular view.

```ts
interface Placement {
  target: ViewTarget;
  viewId: ViewId;
  x: number;
  y: number;
  pinned: boolean;
}
```

An embedding may have hundreds of dimensions. A view projects it into two or
three dimensions and may combine it with containment, provenance, user pins,
and layout constraints.

## Spatial memory

Semantic recalculation and corpus growth should not arbitrarily destroy the
user's spatial memory. Therefore placement may need persistence, anchoring,
or a stability term independent of current embedding coordinates.

## Nebulas

A nebula is an emergent, changeable, manipulable view over a field of related
objects and candidates.

Nebulas are not hard-coded structures. They are produced from kernel facts,
indexes, resolver candidates, permissions, lenses, and user interaction.

Example: opening `Janus` as a nebula may show local concepts, mythology,
companies, people, semantic echoes, private matches hidden by permission, and
external knowledge-base entities as distinct source layers.

Principle:

```text
A nebula can show ambiguity without resolving it.
```

Users may manipulate a nebula by promoting candidates, rejecting candidates,
pinning source layers, changing lenses, moving stars, or creating explicit
relations.

## Open questions

- Is placement durable user state or reproducible derived state?
- What does a “home” layout anchor to?
- When may search or filtering move objects instead of illuminating them?
- How do hard relations and soft similarity jointly influence layout?
- Can a chunk have many simultaneous placements in different views?
