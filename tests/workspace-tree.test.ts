import assert from 'node:assert';
import type { IngestionStatus, RepresentationRecord, SourceKind } from '../src/host/ingestion';
import {
  WORKSPACE_ROOT,
  ancestorDirectoryIds,
  containerExists,
  containerForDocument,
  parentContainer,
  workspaceChildren,
  workspaceCrumbs,
} from '../src/client/workspaceTree';
import type { SourceItemView } from '../src/client/useWorkspace';

const source = (
  relPath: string,
  kind: SourceKind,
  status: IngestionStatus,
  docChunkId?: string,
  parentRelPath?: string,
): SourceItemView => {
  const id = `source_${relPath}`;
  const observation = {
    id: `observation_${relPath}`,
    sourceId: id,
    identityKey: relPath.toLowerCase(),
    kind,
    relPath,
    mediaType: kind === 'directory' ? 'inode/directory' : relPath.endsWith('.md') ? 'text/markdown' : 'application/pdf',
    sizeBytes: 1,
    fingerprint: { algorithm: 'sha256' as const, value: relPath.padEnd(64, '0'), basis: kind === 'directory' ? ('directory-entries' as const) : ('file-bytes' as const) },
    symlink: { status: 'not-symlink' as const },
    observedAt: '2026-08-22T00:00:00.000Z',
  };
  const representation: RepresentationRecord | null = docChunkId
    ? {
        id: `representation_${relPath}`,
        sourceId: id,
        observationId: observation.id,
        relationship: 'native',
        mediaType: observation.mediaType,
        adapter:
          relPath === 'notes/alpha.md'
            ? {
                id: 'test',
                version: '1',
                provider: { identity: 'converter-provider', implementationVersion: '2026.08' },
              }
            : { id: 'test', version: '1' },
        rootChunkId: docChunkId,
        contentChunkIds: [`leaf_${relPath}`],
        outputRevisionIds: [`revision_${relPath}`],
        operationIds: [`operation_${relPath}`],
        warnings: [],
        createdAt: observation.observedAt,
      }
    : null;
  return {
    source: { id, identityKey: observation.identityKey, currentObservationId: observation.id, currentRelPath: relPath },
    observation,
    representation,
    lastResult: {
      status,
      observation,
      adapter: representation?.adapter ?? null,
      diagnostics: status === 'failed' ? [{ code: 'test.failed', severity: 'error', phase: 'adapt', message: 'converter unavailable' }] : [],
      representation,
    },
    parentSourceId: parentRelPath ? `source_${parentRelPath}` : null,
    name: relPath === '.' ? '.' : relPath.slice(relPath.lastIndexOf('/') + 1),
    isWorkspaceRoot: relPath === '.' && kind === 'directory',
    presence: 'present',
  };
};

const sources = [
  source('.', 'directory', 'unchanged'),
  source('notes', 'directory', 'unchanged'),
  source('notes/deep', 'directory', 'unchanged', undefined, 'notes'),
  source('notes/alpha.md', 'file', 'imported', 'chunk_alpha', 'notes'),
  source('notes/manual.pdf', 'file', 'unsupported', undefined, 'notes'),
  source('notes/deep/beta.md', 'file', 'unchanged', 'chunk_beta', 'notes/deep'),
  source('broken.txt', 'file', 'failed'),
];

assert.deepEqual(
  workspaceChildren(sources, WORKSPACE_ROOT).map((node) => [node.kind, node.path, node.status]),
  [
    ['directory', 'notes', 'unchanged'],
    ['file', 'broken.txt', 'failed'],
  ],
);
assert.deepEqual(
  workspaceChildren(sources, 'source_notes').map((node) => [node.kind, node.path, node.docChunkId]),
  [
    ['directory', 'notes/deep', undefined],
    ['file', 'notes/alpha.md', 'chunk_alpha'],
    ['file', 'notes/manual.pdf', undefined],
  ],
);
assert.equal(workspaceChildren(sources, 'source_notes').find((node) => node.status === 'unsupported')?.mediaType, 'application/pdf');
assert.equal(
  workspaceChildren(sources, 'source_notes').find((node) => node.path === 'notes/alpha.md')?.adapterLabel,
  'test@1 via converter-provider@2026.08',
);
assert.deepEqual(
  workspaceChildren(sources, WORKSPACE_ROOT).find((node) => node.status === 'failed')?.diagnostics,
  ['converter unavailable'],
);
assert.equal(parentContainer(sources, 'source_notes/deep'), 'source_notes');
assert.deepEqual(workspaceCrumbs(sources, 'source_notes/deep', 'Headspace'), [
  { containerId: WORKSPACE_ROOT, label: 'Headspace' },
  { containerId: 'source_notes', label: 'notes' },
  { containerId: 'source_notes/deep', label: 'deep' },
]);
assert.ok(containerExists(sources, 'source_notes/deep'));
assert.equal(containerExists(sources, 'missing'), false);
assert.equal(
  containerForDocument(sources, [{ docChunkId: 'chunk_beta', relPath: 'notes/deep/beta.md' }], 'chunk_beta'),
  'source_notes/deep',
);
assert.equal(containerForDocument(sources, [], 'unbound'), null);
assert.deepEqual(
  [...ancestorDirectoryIds(sources, new Set(['chunk_beta']))].sort(),
  ['source_notes', 'source_notes/deep'],
  'a deep document hit illuminates every directory portal on its route',
);
const missingAlpha = {
  ...sources.find((item) => item.observation.relPath === 'notes/alpha.md')!,
  presence: 'missing' as const,
  lastResult: null,
};
const missingNode = workspaceChildren(
  sources.map((item) => (item.source.id === missingAlpha.source.id ? missingAlpha : item)),
  'source_notes',
).find((node) => node.sourceId === missingAlpha.source.id)!;
assert.equal(missingNode.status, 'missing');
assert.equal(missingNode.docChunkId, 'chunk_alpha', 'missing source keeps its recoverable representation');
assert.match(missingNode.diagnostics[0], /not present in the latest ingestion run/);

console.log('workspace tree OK — durable sources form navigable containers and document routes');
