# Information shapes

## Purpose

Concrete information types act as architectural fixtures. They test whether the
kernel, adapters, decomposition rules, and views can represent real material
without adding domain-specific concepts to the universal core.

An adapter decides how an external form is interpreted. The kernel records the
result using occurrences, derivations, and links. A view chooses the most useful
projection for a task.

## LLM conversations

An LLM conversation combines several structures:

- messages appear in a readable sequence;
- a message may have several alternative replies;
- a generated message derives from a prompt and selected context;
- a conversation may fork while retaining a shared prefix;
- an imported provider export may expose only one path or the entire branch
  graph.

Possible mapping:

```text
ordered transcript -> message occurrences or a projected reply path
reply topology      -> links with role "replies-to"
generated output    -> provenance plus derivation from input revisions
alternative reply   -> sibling outgoing reply link
```

Accepted mapping:

- the message DAG is canonical truth;
- `replies-to` links target the specific parent revision that received the
  reply;
- generation provenance records the complete input revision set, which may be
  larger than the visible reply path;
- each readable transcript or named branch is a projected or saved path through
  the graph;
- editing a historical sent message to create another continuation is a fork,
  preserving the original branch.

The default interface remains tree-shaped even though the underlying truth is a
graph.

## Articles

An article combines ordered blocks with heterogeneous and inline material:

```text
article
  heading
  paragraph containing an inline link
  image
  paragraph
  figure caption
```

Possible mapping:

- the article container orders paragraph, heading, image, and figure
  occurrences;
- images are chunks or bound external media with their own revisions;
- an authored hyperlink is a link from a source span to a chunk, revision, or
  external address;
- the source HTML or Markdown blob remains available for faithful round-trip;
- an adapter or decomposer exposes block and inline structure without requiring
  every token to become a chunk.

This fixture shows that block order, inline addresses, content identity, and
external binding are separate concerns.

## Questions exposed by the fixtures

- Does `Position` need hierarchical block and inline coordinates rather than a
  single numeric index?
- Which explicit structures from an imported document are promoted immediately,
  and which remain derived parser facts?
- How does an adapter preserve faithful source order while allowing promoted
  chunks to be rearranged?
