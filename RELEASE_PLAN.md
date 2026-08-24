# Headspace 0.1.0 release workback

This is a living implementation plan derived backward from [RELEASE_NOTES.md](RELEASE_NOTES.md). The release notes describe the promised experience; this document records what must become true to make that promise honest.

## Product proof

Version 0.1.0 proves one loop:

> ingest → navigate → focus → transform → review → integrate → restart

A capability is not complete merely because its kernel operation or endpoint exists. It is complete when a new user can encounter it through the released application, understand what happened, recover from failure, and verify the result after restart.

## Working rules

- Complete one bounded loop at a time, with its acceptance test written before or alongside implementation.
- Keep the kernel stable unless a product loop demonstrates a missing invariant.
- Put providers, parsers, stores, indexes, and external effects behind explicit seams rather than adding their dependencies to the kernel.
- Give every seam one honest reference implementation before generalizing it.
- Preserve original source identity. A conversion is a versioned derivation, not the source itself.
- Never silently overwrite external work or automatically apply generated work.
- Keep unsupported inputs and failures visible rather than dropping them from the workspace.
- Treat the repository's existing `package.json` and `wiki/resolver.md` edits as user-owned work.

## Baseline established on 2026-08-21

- All six automatic test suites pass.
- Strict TypeScript checking and the Vite production asset build pass.
- The kernel, filesystem store, Markdown reconciliation, merge behavior, and lexical indexes have substantial automated coverage.
- The current UI demonstrates navigation into a document, editing, raw-span promotion, deterministic dispatch, proposal acceptance/rejection, and explicit Markdown projection.
- The current development server is fixed to the repository's own documentation corpus.
- The production asset build has no production host for its API.
- Ingestion is directly coupled to the Markdown driver; directory scans ignore non-Markdown files.
- Directory hierarchy is inferred from path strings rather than represented as navigable containers.
- Dispatch uses the deterministic stub, selected context is not inspectable, and completed proposal outcomes disappear from the UI.

## Ordered release loops

### Loop 0 — Define the release

Status: complete

Deliverables:

- Draft the aspirational release notes.
- Separate the product promise from the long-term wiki backlog.
- Establish passing test, typecheck, and build baselines.
- Record explicit non-goals for 0.1.0.

### Loop 1 — Establish the safety floor

Status: complete

Goal: existing source bindings must not damage or silently misidentify external material while later ingestion work expands their reach.

Acceptance:

- Projection refuses to overwrite a source whose on-disk fingerprint has changed since the last observed or projected state.
- Binding/sidecar replacement is atomic and a failed replacement does not destroy the previous binding.
- A scanned symlink cannot cause ingestion or projection outside the selected root.
- Sampled or ambiguous block matches never silently inherit identity without an explicit confidence-bearing review path.
- Tests cover each refusal and recovery path.

Verified:

- Projection fingerprint conflicts and missing sources return explicit refusal without changing source or sidecar.
- Projection and sidecar writes use same-directory atomic replacement; injected publish failure preserves the previous file.
- Lexical traversal, file symlinks, and directory junctions cannot escape the workspace during scan or projection.
- Sampled and ambiguous similarity matches raise proposals containing their confidence instead of entering the automatic fast path.
- Eleven automatic test files and the production build pass after the changes.

### Loop 2 — Make ingestion an observable seam

Status: complete

Goal: turn the hard-coded Markdown sweep into a small adapter pipeline without attempting to build every converter.

Acceptance:

- A root-confined directory source emits explicit file and directory observations.
- Each observation records source identity, relative path, media type, size, fingerprint, and symlink status.
- An ingestion adapter declares its ID, version, accepted media types, and output behavior.
- Native Markdown and plain-text adapters are separate reference tenants.
- Each item reports `imported`, `updated`, `unchanged`, `proposal`, `unsupported`, or `failed` with adapter and diagnostic information.
- A second scan and process restart preserve source identities.

Verified:

- Directory, file, symlink, unsupported-media, and decode-failure observations remain visible in a durable workspace catalog.
- Native Markdown and plain-text adapters declare accepted inputs, exact outputs, write-back behavior, and versioned identities; undeclared products are refused.
- Source identity is canonical across overlapping inputs and Windows path casing, while configured lexical escapes and resolved symlink/junction escapes are refused.
- Initial materialization uses a write-ahead intent correlated to the exact kernel import operation. Fault-injected catalog and Markdown-sidecar publication failures recover through real log replay without duplicating chunks or changing source identity.
- Representation provenance names the exact operations and output revisions that produced it rather than whichever operation happened to run last.
- A newer external observation supersedes older source proposals before adapter selection, including when the new bytes are unreadable; stale external work cannot later be accepted.
- Fourteen automatic test files, strict TypeScript checking, and the production build pass after the completed Loop 2 changes.

### Loop 3 — Make the workspace navigable at multiple scales

Status: complete

Goal: expose the ingested shape rather than flattening it into document stars grouped by path text.

Acceptance:

- A user can open or configure one local workspace and see which root is active.
- Nested directories are real navigable containers.
- Navigation supports workspace → container → document → addressable part and a clear route back.
- Unsupported and failed sources remain visible with their status.
- Existing Star editing and Nebula orientation continue to work.

Verified:

- The API projects durable directory observations into stable `SourceId` parent edges; the client never has to invent hierarchy from display paths.
- The persistent workspace header shows the active name and root, supports stable-ID breadcrumbs, and returns from Star to the originating container.
- Directory portals expose nested skies while represented documents remain editable stars and unbound substrate documents remain visible at workspace scale.
- Unsupported sources, failed refreshes, and missing sources remain visible and inspectable. A retained last-good representation stays usable without disguising its external status.
- Deep document search hits illuminate every ancestor directory portal without rearranging the sky.
- A `HEADSPACE_WORKSPACE` startup setting selects one local root; the repository design corpus remains the zero-configuration demo.
- Plain text correctly hides Markdown-only projection, while existing Star editing, proposal badges, provenance lens, and addressable part controls remain wired.
- Hierarchy, delete/restart presence, failed-refresh retention, source status, stable routing, and rendered navigation have automatic coverage.

### Loop 4 — Make collaboration inspectable end to end

Status: complete

Goal: make the existing deterministic collaborator a truthful product feature before adding a hosted model.

Acceptance:

- The UI identifies the active collaborator as the offline deterministic adapter.
- Before dispatch, the bounded context is visible with item identities, revisions, roles, and inclusion reasons.
- The proposal inspector shows author, inputs, basis revisions, targets, all textual and structural changes, and current freshness.
- Accepted, rejected, stale, and superseded proposals remain visible as history.
- Provider or transport failure creates no partial truth.
- Accepted work survives a host restart with author, acceptor, inputs, derivations, and proposal outcome intact.

Verified:

- The deterministic collaborator is a named tenant outside the kernel; dispatch requires an explicit completer and author identity.
- The exact bounded context is visible before dispatch with roles, inclusion reasons, chunk, occurrence, and revision identities, including recursively rendered and pinned dependencies.
- Text heads, redaction/tombstone state, occurrence order, placement, pinning, severing, and nested rendered dependencies are captured before provider latency and participate in proposal freshness.
- Provider failure occurs before proposal creation and leaves the head, commits, operations, chunks, and proposal set unchanged.
- Proposal history retains open, accepted, rejected, stale, and superseded work. Its inspector uses immutable basis text and always exposes the exact textual and structural payload, author, dispatcher, inputs, producer, targets, resolution operation, and reason.
- Acceptance provenance—including proposer, acceptor, input revisions, derivation source/operation, and outcome—survives log replay and restart.
- Delayed-provider adversarial tests cover focus and child edits, nested edits, redaction, reorder, sever, pinned leaf, and pinned composite semantics.

### Loop 5 — Prove replaceability with external tenants

Status: complete

Goal: demonstrate that richer sources and real intelligence plug into seams rather than into the kernel.

Acceptance:

- One converter-backed document adapter produces a derived representation while retaining original source identity, adapter version, and warnings.
- One real model adapter is configured outside the kernel and outside source control secrets.
- Stub and real adapters expose the same capability and failure contract.
- Converter or model unavailability degrades visibly without preventing offline use.

Verified:

- The optional HTTP PDF-to-Markdown tenant retains the PDF source identity and records its exact observation, adapter, provider identity/version, derived output revisions, operations, and warnings. Its endpoint and bearer credential never enter durable state.
- Converter work is bounded by an end-to-end deadline, response-size cap, fatal UTF-8/JSON validation, redirect refusal, HTTPS credential rule, and a per-run retryable-failure circuit. Failure creates no partial representation or source materialization.
- Write-ahead recovery binds initial imports and revisions to the exact adapter actor, product, source text, rendered text, output revisions, and composite child structure. Tampered or ambiguous historical work remains visibly pending rather than acquiring false provenance, while later human heads are preserved.
- The optional OpenAI Responses tenant is configured only when both an API key and explicit model are present on the host. The browser sends an allowlisted collaborator ID, visible bounded context, and instruction; credentials stay host-side.
- Remote proposal provenance records the configured capability, provider-resolved model, and response receipt. Provider, transport, timeout, malformed-response, or status failure creates no proposal or kernel truth.
- The deterministic local collaborator remains available with no external configuration, and unavailable external tenants remain visible with actionable diagnostics.

### Loop 6 — Ship a release-shaped application

Status: implementation complete; release operations pending

Goal: make the proven loop usable without the repository-specific Vite development fixture.

Acceptance:

- One documented command starts the built client and durable host APIs.
- An automated release-runtime test performs the complete product loop, stops the host, restarts it, and verifies integrated state and provenance.
- Browser-side pending work is durable across reload, or reload is explicitly prevented until the host acknowledges it.
- A clean checkout can install, test, typecheck, and build in CI using a declared Node version.
- Headspace/Substrate naming is deliberately resolved.
- Package and lockfile versions agree; README, license decision, supported-platform statement, final release notes, and `v0.1.0` tag are ready.

Verified:

- `npm start` serves the production client and durable APIs from one loopback-only Node host after eagerly opening, locking, replaying, and ingesting the selected workspace.
- The runtime confines its workspace and static roots, rejects non-loopback binding, foreign request authorities and API origins, traversal and symlink escapes, and applies release-wide framing, referrer, and content-type protections.
- The automated runtime test performs mixed ingestion, bounded selection, deterministic dispatch, proposal review and acceptance, host shutdown, fresh restart, and durable authorship, acceptance, input, derivation, and text checks.
- Browser commits use a single-flight acknowledgement queue and unload warning. Reload/ingest waits for active dispatch and pending commits; divergence or an ambiguous mutating response quarantines the browser until a successful authoritative reload.
- Node 22.12 is the declared minimum; package and lockfile both name `headspace` 0.1.0; CI is configured for Windows and Ubuntu; isolated copies on both platforms pass `npm ci`, the advisory audit, and the full test/typecheck/build gate. A live Vite 8 development host also serves the shell and substrate API correctly.
- Headspace is the product name and Substrate is its internal kernel/store. The README records setup, data location, external egress, supported platforms, limitations, and the deliberate `UNLICENSED` decision.

### Post-0.1 hardening queue

These audit findings do not block the trusted, loopback-only 0.1 contract, but should remain visible:

- Make multi-commit HTTP batches atomic rather than admitting their commits sequentially.
- Extend hostile-JSON shape validation uniformly across every generic fact collection, beyond the strict envelope and proposal paths required for this release.
- Decide whether trusted external tenants may return arbitrary durable text/warnings, or whether future adapters need provider-output secret scanning and redaction.
- Bound shutdown time for active HTTP requests before a future packaged or remotely managed host depends on graceful termination.
- Move the Vite configuration dependency graph to explicit TypeScript extensions before Vite's native config loader becomes the default; 0.1.0 deliberately uses the current bundled loader.

## Release gate

Do not tag 0.1.0 until every acceptance statement above is either verified or deliberately removed from the release contract. A passing kernel suite alone is necessary but not sufficient; the released user loop is the product.

Local implementation gate, 2026-08-22: **passed** — isolated Windows and Ubuntu copies pass `npm ci`, report zero known npm vulnerabilities, and pass 25/25 test files, strict TypeScript checking, and the Vite 8 production build; the source tree also passes an adversarial release audit and a documented-command restart smoke test.

Release operations still required before the tag: commit the dirty/untracked release tree without absorbing unrelated user work, connect a Git remote, let the Windows and Ubuntu CI jobs pass from that commit, then create `v0.1.0`. No remote or tag is currently configured. The authenticated account's existing public `asavs/headspace` repository contains a different legacy Python application and unrelated history, so this workback will not attach or overwrite it by inference.
