# Headspace 0.0.1

Headspace 0.0.1 is an intentionally narrow preview of the local text kernel. It proves that one local workspace can be opened, navigated, edited through a versioned graph, and recovered after restart without silently replacing the user's source files.

## Release contract

- Select one local workspace when the host starts.
- Discover and navigate directories containing Markdown and UTF-8 plain-text documents.
- Open a document and edit its representation through the versioned workspace graph.
- Persist graph state under the workspace's `.headspace/` directory and recover it after a host restart.
- Project an edited Markdown representation back to its source only through an explicit user action and only when the safety checks pass.
- Keep the original filesystem source authoritative. An in-graph edit is not an implicit source-file write.

Before Markdown projection, Headspace checks that the source still matches the version it observed. If the source changed, disappeared, or cannot be safely addressed inside the workspace, projection is refused instead of overwriting it. Plain-text source projection is not part of the 0.0.1 contract.

## Preview boundaries

The repository contains work beyond this contract. In 0.0.1, advanced ingestion and conversion, AI or model collaboration, complete spatial relationship and layout semantics, and a universal ingest-to-integration product loop are experimental. Their presence in code or the interface is not a compatibility or support promise.

This preview is local, single-workspace, single-user, and loopback-only. Its persisted schema is pre-release data; back up source material and do not assume future versions will migrate `.headspace/` state.

## Run it

Headspace 0.0.1 requires Node.js 22.12 or newer.

```powershell
npm ci
npm run build
$env:HEADSPACE_WORKSPACE = 'C:\path\to\a\small-workspace'
npm start
```

On macOS or Linux, use `export HEADSPACE_WORKSPACE=/path/to/workspace`. Open the loopback URL printed by the host. See [README.md](README.md) for operating details and [RELEASE_PLAN.md](RELEASE_PLAN.md) for the release gate.

The broader 0.1.0 direction remains available as explicitly non-current drafts in [`docs/releases/`](docs/releases/).
