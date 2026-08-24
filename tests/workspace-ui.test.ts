import assert from 'node:assert';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import type { SubstrateHook } from '../src/App';
import { Nebula, SourceStatusPanel } from '../src/Nebula';
import { Star } from '../src/Star';
import { workspaceChildren, WORKSPACE_ROOT } from '../src/client/workspaceTree';
import { workspacePayload } from '../src/host/api';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';
import { syncWorkspace } from '../src/host/sync';

const root = mkdtempSync(join(tmpdir(), 'headspace-workspace-ui-'));
let ws: WorkspaceStore | null = null;
try {
  mkdirSync(join(root, 'sources'), { recursive: true });
  writeFileSync(join(root, 'sources', 'note.txt'), 'A visible note');
  writeFileSync(join(root, 'sources', 'warned.txt'), 'A represented source with a warning');
  writeFileSync(join(root, 'sources', 'manual.pdf'), Buffer.from('%PDF-not-adapted'));
  ws = await openWorkspace(root);
  await syncWorkspace(ws, { contentDirs: ['sources'] });
  writeFileSync(join(root, 'sources', 'note.txt'), Buffer.from([0xc3, 0x28]));
  await syncWorkspace(ws, { contentDirs: ['sources'] });
  const payload = workspacePayload(root, ws);
  const warned = payload.sources.find((source) => source.observation.relPath === 'sources/warned.txt')!;
  warned.lastResult!.diagnostics.push({
    code: 'adapter.example-warning',
    severity: 'warning',
    phase: 'adapt',
    message: 'The adapter retained a recoverable conversion warning',
  });
  const sub = {
    ws: {
      state: ws.state,
      bindings: payload.bindings,
      identity: payload.workspace,
      adapters: payload.adapters,
      sources: payload.sources,
      lastIngestion: payload.lastIngestion,
    },
    ctx: ws.ctxFor('human:test'),
    version: 1,
    error: null,
    status: null,
    busy: false,
    ingestNow: async () => null,
    syncNow: async () => null,
    reload: async () => null,
    dismissStatus: () => undefined,
  } as unknown as SubstrateHook;

  const rootMarkup = renderToStaticMarkup(
    createElement(Nebula, {
      sub,
      containerId: WORKSPACE_ROOT,
      onOpenContainer: () => undefined,
      onFocus: () => undefined,
    }),
  );
  assert.match(rootMarkup, /Open sources container/);
  assert.doesNotMatch(rootMarkup, /manual\.pdf/, 'deep descendants do not flatten into the workspace sky');

  const sourcesDirectory = payload.sources.find((source) => source.observation.relPath === 'sources')!;
  const directoryMarkup = renderToStaticMarkup(
    createElement(Nebula, {
      sub,
      containerId: sourcesDirectory.source.id,
      onOpenContainer: () => undefined,
      onFocus: () => undefined,
    }),
  );
  assert.match(directoryMarkup, /Open note\.txt; source refresh failed/);
  assert.match(directoryMarkup, /Inspect source status for note\.txt/);
  assert.match(directoryMarkup, /Open warned\.txt; source has an ingestion warning/);
  assert.match(directoryMarkup, /Inspect source status for warned\.txt/);
  assert.match(directoryMarkup, /Inspect manual\.pdf, unsupported/);

  const issue = workspaceChildren(payload.sources, sourcesDirectory.source.id).find(
    (node) => node.status === 'unsupported',
  )!;
  const issueMarkup = renderToStaticMarkup(
    createElement(SourceStatusPanel, {
      source: issue,
      busy: false,
      onRetry: () => undefined,
      onClose: () => undefined,
    }),
  );
  assert.match(issueMarkup, /application\/pdf/);
  assert.match(issueMarkup, /no adapter available/);
  assert.match(issueMarkup, /No available adapter accepts application\/pdf/);

  const providerMarkup = renderToStaticMarkup(
    createElement(SourceStatusPanel, {
      source: {
        ...issue,
        adapterLabel: 'headspace.pdf-to-markdown.http@1.0.0 via tenant-a@converter-2026.08',
      },
      busy: false,
      onRetry: () => undefined,
      onClose: () => undefined,
    }),
  );
  assert.match(
    providerMarkup,
    /headspace\.pdf-to-markdown\.http@1\.0\.0 via tenant-a@converter-2026\.08/,
  );

  const failedRefresh = workspaceChildren(payload.sources, sourcesDirectory.source.id).find(
    (node) => node.status === 'failed' && node.docChunkId,
  )!;
  const failedMarkup = renderToStaticMarkup(
    createElement(SourceStatusPanel, {
      source: failedRefresh,
      busy: false,
      onRetry: () => undefined,
      onClose: () => undefined,
    }),
  );
  assert.match(failedMarkup, /headspace\.text\.native@1\.0\.0/);
  assert.match(failedMarkup, /encoded data|UTF-8|utf-8/i);

  const textBinding = payload.bindings.find((binding) => binding.relPath === 'sources/note.txt')!;
  const starMarkup = renderToStaticMarkup(
    createElement(Star, {
      sub,
      docId: textBinding.docChunkId,
      onFocusDoc: () => undefined,
      onBack: () => undefined,
      backLabel: 'sources',
    }),
  );
  assert.match(starMarkup, /← sources/);
  assert.match(starMarkup, /A visible note/, 'a failed refresh retains the last good representation');
  assert.doesNotMatch(starMarkup, /project → file/, 'plain text does not advertise unsupported write-back');

  console.log('workspace UI OK — nested skies, visible source failures, and routed Star return');
} finally {
  ws?.close();
  rmSync(root, { recursive: true, force: true });
}
