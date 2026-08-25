# Headspace

Headspace 0.1.0 is an intentionally narrow preview of a local text kernel. It opens one local workspace, represents Markdown and UTF-8 plain text in a versioned workspace graph, and keeps graph edits durable across host restarts.

The preview contract is deliberately small: navigate, open, edit through the graph, persist, restart, and explicitly project Markdown when it is safe. The original filesystem source remains authoritative.

## Quick start

Headspace requires Node.js 22.12 or newer and npm. Windows 11 is the primary supported platform; CI also verifies current Ubuntu runners.

```powershell
npm ci
npm run build
$env:HEADSPACE_WORKSPACE = 'C:\path\to\a\small-workspace'
npm start
```

On macOS or Linux, set the workspace with:

```sh
export HEADSPACE_WORKSPACE=/path/to/a/small-workspace
npm start
```

Open the loopback URL printed by the host. If `HEADSPACE_WORKSPACE` is absent, the host uses its current directory. Start with a copy or a disposable workspace while evaluating this pre-release build.

For UI development, run `npm run dev`. The development fixture is convenient for interface work, but `npm run build` followed by `npm start` is the release-shaped path.

## What 0.1.0 promises

- One local workspace selected when the host starts.
- Navigation through its directories and opening Markdown or UTF-8 plain-text documents.
- Edits recorded as versioned workspace-graph state.
- Durable state under `<workspace>/.headspace/` that is recovered after restart.
- Explicit, conflict-checked projection from an edited Markdown representation to its bound Markdown source.
- No automatic overwrite of the original source.

Editing the graph does not edit the source file. Markdown changes reach the filesystem only when the user explicitly requests projection. Before writing, Headspace verifies that the source still matches the observed version and refuses the operation if it changed, disappeared, or falls outside the selected workspace. Plain-text projection is not supported by the 0.1.0 contract.

The original file is the authoritative external source. `.headspace/` contains the preview's log, snapshots, source catalog, and Markdown identity sidecars; it is application state, not a replacement for source backups. The persisted format is pre-release and carries no forward-migration guarantee.

## Experimental surfaces

The codebase also contains advanced ingestion and conversion, collaborator and proposal flows, and richer spatial concepts. For 0.1.0 these are experimental—not supported release promises. In particular, the preview does not promise:

- PDF or universal-format ingestion and conversion
- AI or hosted-model collaboration
- a complete spatial relationship or layout model
- a universal ingest → orient → focus → collaborate → review → integrate loop
- multiple workspaces, multi-user collaboration, or network hosting

Experimental capabilities may require additional configuration or external services. Their presence does not expand the 0.1.0 contract.

## Host configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `HEADSPACE_WORKSPACE` | Local workspace root | Current directory |
| `HEADSPACE_HOST` | Loopback bind address (`127/8`, `localhost`, or `::1`) | `127.0.0.1` |
| `HEADSPACE_PORT` | HTTP port | `4173` |
| `HEADSPACE_DIST` | Built client directory | `./dist` |

The HTTP host has no authentication, so 0.1.0 is deliberately loopback-only and rejects foreign `Host` and `Origin` authorities. Do not expose it to a network.

## Verify the preview

```powershell
npm run verify
```

That runs the dependency audit, automated tests, strict TypeScript checking, and production build. Tests may cover experimental internals; passing them does not add those surfaces to the release contract.

The current contract and workback live in [RELEASE_NOTES.md](RELEASE_NOTES.md) and [RELEASE_PLAN.md](RELEASE_PLAN.md). The recovered project history is mapped in [LINEAGE.md](LINEAGE.md) and told in [The Headspace Lineage](docs/the-headspace-lineage.md).

## License

No license is granted for this 0.1.0 source preview (`UNLICENSED`). Choose and add an appropriate license before redistributing the project as open source.
