# Headspace

Headspace is a local spatial environment that brings heterogeneous material into a versioned workspace graph, makes it navigable at several scales, and lets people or model collaborators propose changes without silently overwriting the source.

Version 0.1.0 proves one complete loop:

> ingest → navigate → focus → transform → review → integrate → restart

**Headspace** is the product. A shared **kernel** enforces the workspace graph's invariants, while the local **host** owns authoritative operations and a replaceable **store** makes them durable.

## Quick start

Headspace 0.1.0 requires Node.js 22.12 or newer and npm. Windows 11 is the primary supported platform; CI is configured to verify both Windows and current Ubuntu runners.

```powershell
npm ci
npm run build
$env:HEADSPACE_WORKSPACE = 'C:\path\to\a\small-workspace'
npm start
```

On macOS or Linux, set the workspace with `export HEADSPACE_WORKSPACE=/path/to/workspace`. Open the loopback URL printed by the host. If `HEADSPACE_WORKSPACE` is absent, the release host scans its current directory.

Use a small copy or test workspace first. Headspace keeps its append-only log, snapshots, ingestion catalog, and sidecars in `<workspace>/.headspace/`. The selected filesystem remains authoritative in 0.1.0; source write-back is never automatic and is offered only where the active adapter declares a safe projection.

For UI development, run `npm run dev`. With no selected workspace, the development fixture opens this repository's design corpus; the built release host instead defaults to its current directory.

## What 0.1.0 ingests

- Directories become stable, navigable containers.
- Markdown and UTF-8 plain text use native local adapters.
- PDF is an optional derived representation supplied by a replaceable HTTP converter.
- Unsupported, missing, and failed sources stay visible with diagnostics; they do not disappear from the workspace.

A conversion never replaces the source's identity. Headspace records the exact observation, adapter, provider implementation, output revisions, operations, and warnings behind the derived representation.

### PDF converter seam

Configure all three non-secret identity fields plus the service URL:

```powershell
$env:HEADSPACE_PDF_CONVERTER_URL = 'https://converter.example/v1/pdf-to-markdown'
$env:HEADSPACE_PDF_CONVERTER_SERVICE_IDENTITY = 'example.pdf-service'
$env:HEADSPACE_PDF_CONVERTER_IMPLEMENTATION_VERSION = '2026-08-01'
$env:HEADSPACE_PDF_CONVERTER_TOKEN = 'replace-with-a-secret-if-required'
```

Headspace sends `POST` with raw `application/pdf` bytes and accepts an `application/json` response:

```json
{
  "mediaType": "text/markdown",
  "text": "# Converted document",
  "warnings": ["optional conversion warning"]
}
```

Bearer credentials require HTTPS, except for an explicit loopback host. The default deadline is 30 seconds and the response cap is 8 MiB; override them with `HEADSPACE_PDF_CONVERTER_TIMEOUT_MS` and `HEADSPACE_PDF_CONVERTER_MAX_RESPONSE_BYTES`. An outage opens a circuit for the remainder of that ingestion run so a folder of PDFs cannot multiply one provider failure indefinitely.

## Collaborators

The offline deterministic collaborator is always available and makes the proposal workflow testable without a network or account. To expose the optional OpenAI Responses collaborator, set both variables in the host environment before starting Headspace:

```powershell
$env:OPENAI_API_KEY = 'replace-with-a-secret'
$env:HEADSPACE_OPENAI_MODEL = 'a-model-available-to-your-account'
```

Headspace has no hidden default model. The API key stays in the host process; the browser can choose only a capability the host has exposed. Dispatching to a remote collaborator sends the visible bounded context and instruction to that provider. The UI labels that egress before dispatch, and a provider or transport failure creates no proposal or partial kernel state.

A successful remote proposal records the exposed collaborator identity and version, the model identity returned by the provider, and the provider response receipt. Credentials and endpoint secrets are never written into workspace graph state.

Secrets are ignored by Git. `.env.example` documents variable names, but 0.1.0 does not automatically load `.env`; inject values through your shell or service environment.

## Host configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `HEADSPACE_WORKSPACE` | Local workspace root | Current directory |
| `HEADSPACE_HOST` | Loopback bind address (`127/8`, `localhost`, or `::1`) | `127.0.0.1` |
| `HEADSPACE_PORT` | HTTP port | `4173` |
| `HEADSPACE_DIST` | Built client directory | `./dist` |
| `OPENAI_API_KEY` | Optional hosted-model credential | Unavailable |
| `HEADSPACE_OPENAI_MODEL` | Explicit OpenAI model ID | Unavailable |
| `HEADSPACE_PDF_CONVERTER_URL` | Optional PDF converter endpoint | Unavailable |
| `HEADSPACE_PDF_CONVERTER_SERVICE_IDENTITY` | Durable, non-secret converter identity | Required with URL |
| `HEADSPACE_PDF_CONVERTER_IMPLEMENTATION_VERSION` | Durable converter implementation version | Required with URL |
| `HEADSPACE_PDF_CONVERTER_TOKEN` | Optional converter bearer secret | None |

The HTTP host has no authentication in 0.1.0, so it is deliberately loopback-only and rejects foreign `Host` and `Origin` authorities. Network sharing requires a future authenticated host. Static assets are served only from the canonical built directory, and source observation and projection remain confined to the canonical workspace root.

## Verify the release

```powershell
npm run verify
```

That checks npm's current advisories at moderate severity or higher, then runs the automatic suite, strict TypeScript checking, and the production build. The release-runtime test serves the client and APIs from one host, exercises mixed ingestion and the collaboration proposal contract, restarts a fresh host, and checks the integrated text and provenance.

The release promise and its workback live in [RELEASE_NOTES.md](RELEASE_NOTES.md) and [RELEASE_PLAN.md](RELEASE_PLAN.md).

## Current boundaries

Headspace 0.1.0 selects one workspace at host startup; it does not yet include an in-app folder picker or multi-root workspaces. It also does not move files by spatial drag-and-drop or automatically reorganize the filesystem.

This release does not promise first-party parsing for every format, autonomous changes, continuous bidirectional sync, multi-user realtime collaboration, durable spatial coordinates, semantic clustering, external-web snapshotting, or a plugin marketplace. Those are future implementations behind the same seams, not prerequisites hidden inside this kernel.

## License

No license is granted for this 0.1.0 source release (`UNLICENSED`). That is a deliberate release decision, not an accidental omission; choose and add an open-source license before redistributing the project as open source.
