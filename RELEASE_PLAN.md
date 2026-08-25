# Headspace 0.1.0 release workback

This plan works backward from the narrow contract in [RELEASE_NOTES.md](RELEASE_NOTES.md). Version 0.1.0 is a text-kernel preview, not the broader product loop.

## Acceptance proof

Use a fresh, disposable workspace containing at least one Markdown file, one UTF-8 plain-text file, and a nested directory. The release candidate must demonstrate:

1. The built host opens exactly the selected local workspace.
2. The user can navigate the directory structure and open both text documents.
3. A document edit creates versioned graph state without changing the original source file.
4. Stopping and restarting the host preserves the in-graph edit.
5. An explicit Markdown projection writes the accepted graph text to the bound Markdown source.
6. Projection refuses to overwrite a source changed externally after observation.
7. A second restart preserves the integrated graph state and its source binding.

The proof should cover the released, built client and host where practical. Lower-level tests remain necessary for graph, persistence, confinement, and projection invariants.

## Safety gate

- Source writes occur only after an explicit Markdown projection request.
- The original source remains authoritative until that request succeeds.
- Projection checks the observed source version and refuses conflicts, missing sources, traversal, and workspace escapes.
- Plain-text projection is not presented as supported in 0.1.0.
- Durable preview data stays inside `<workspace>/.headspace/`.
- The unauthenticated host remains restricted to loopback.

## Packaging gate

- `package.json` and the root lockfile package agree on version `0.1.0`.
- README and current release documents describe only the 0.1.0 contract.
- A clean install passes tests, strict type checking, and the production build on Windows and Ubuntu using the declared Node version.
- `npm start` serves the built client and authoritative local APIs for the selected workspace.
- The exact publication commit passes CI before the `0.1.0` tag is created.

## Explicitly outside the gate

Advanced ingestion and converters, AI collaboration, complete spatial relations, and the universal ingest → orient → focus → collaborate → review → integrate loop do not block 0.1.0 and are not promised by it. They may remain visible as experimental work, but current documentation must not imply release support.

Those broader capabilities remain future work behind the same seams. They are not hidden requirements for this release.

## Release decision

Do not publish 0.1.0 until every acceptance and safety statement above has executable evidence or has been removed from the contract. Passing internal kernel tests alone does not prove the built preview experience.

## Current gate status

Local automated gate, 2026-08-24: **passed**. `npm run verify` reports zero known npm vulnerabilities, 26/26 passing test files, strict TypeScript success, and a successful production build.

The release-runtime proof opens a real disposable host workspace, ingests Markdown and plain text, commits a versioned text edit through the host boundary, projects Markdown explicitly, refuses an externally changed source without altering its bytes, restarts a fresh host, and verifies the edit, prior revision, ancestry, authorship, operation provenance, plain-text representation, and binding state.

Release operations still pending: publish the exact candidate commit so Windows and Ubuntu CI can verify it, and create `0.1.0` only after those checks pass. The intended Git remote is configured as `origin`.
