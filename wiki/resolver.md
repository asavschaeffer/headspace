# Resolver

## Purpose

The resolver proposes possible referents for a span, chunk, phrase, image,
name, or concept.

It does not decide final identity. It produces candidates with evidence so
Janus, permissions, lenses, views, and users can reason about them.

## Accepted principles

### Mention is not reference

A matching word or name does not prove that the author meant a particular
thing.

```text
mention is not reference
reference is not identity
similarity is not derivation
name match is not meaning
```

The resolver may propose candidates. Confirmation or stronger evidence is
needed before turning a candidate into durable truth.

### A span may open into many referents

Example: `Janus` may connect to:

- the local Substrate threshold-guardian seam;
- `wiki/janus.md`;
- Janus, the Roman god;
- Janus-faced language;
- companies and projects named Janus;
- people or handles named Janus;
- a pseudonymous AI alignment researcher;
- semantic neighbors such as thresholds, gates, identity, and transitions.

The system should not collapse these into one canonical target.

### Candidates carry evidence, not generic confidence

A resolver candidate should explain why it exists.

```ts
interface ReferentCandidate {
  span: TextSpan;
  target: ResourceId | ExternalEntityId;
  kind:
    | "local-concept"
    | "external-entity"
    | "person"
    | "organization"
    | "mythology"
    | "lexical"
    | "semantic-echo"
    | "unknown";
  evidence: Evidence[];
  status: "candidate" | "confirmed" | "rejected";
}
```

Scores may exist inside method-specific evidence, such as embedding distance,
but a generic confidence number should not be the primitive.

## Source layers

Resolver results may come from multiple layers:

```text
personal/private corpus
local project corpus
shared/team corpus
public Substrate corpus
approved knowledge bases
general web
```

Each source layer should remain visible and permission-distinct.

## Relationship to other seams

- Index finds lexical, span, embedding, and entity candidates.
- Resolver packages candidates as possible referents.
- Janus guards identity and boundary claims.
- Permissions filter what can be shown.
- Lenses cluster and interpret candidate fields.
- Views render candidate fields as navigable nebulas.
- Users may promote candidates into explicit references.
