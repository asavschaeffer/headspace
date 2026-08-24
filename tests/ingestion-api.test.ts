import assert from 'node:assert';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { workspacePayload } from '../src/host/api';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';
import { syncWorkspace } from '../src/host/sync';

const root = mkdtempSync(join(tmpdir(), 'headspace-ingestion-api-'));
let ws: WorkspaceStore | null = null;
try {
  mkdirSync(join(root, 'sources'), { recursive: true });
  mkdirSync(join(root, 'sources', 'ideas'), { recursive: true });
  writeFileSync(join(root, 'sources', 'readme.md'), '# Read me\n');
  writeFileSync(join(root, 'sources', 'note.txt'), 'A plain note');
  writeFileSync(join(root, 'sources', 'ideas', 'alpha.md'), '# Alpha\n');
  writeFileSync(join(root, 'sources', 'image.bin'), Buffer.from([0, 1, 2]));
  writeFileSync(join(root, 'sources', 'invalid.txt'), Buffer.from([0xc3, 0x28]));
  writeFileSync(join(root, 'loose.md'), '# Loose\n');

  ws = await openWorkspace(root);
  const report = await syncWorkspace(ws, { contentDirs: ['sources'], contentFiles: ['loose.md'] });
  const payload = workspacePayload(root, ws);

  assert.equal(payload.workspace.id?.startsWith('workspace_'), true);
  assert.equal(payload.workspace.rootDisplayPath, root);
  assert.deepEqual(
    payload.adapters.map((adapter) => adapter.id),
    ['headspace.markdown.native', 'headspace.text.native', 'headspace.pdf-to-markdown.http'],
  );
  assert.deepEqual(
    payload.collaborators.map((collaborator) => collaborator.id),
    ['headspace.offline-deterministic', 'openai.responses'],
  );
  assert.equal(payload.collaborators[0].execution, 'local');
  assert.equal(payload.collaborators[1].execution, 'remote');
  assert.doesNotMatch(JSON.stringify(payload.collaborators), /OPENAI_API_KEY\s*[:=]\s*[^".]/);
  assert.equal(payload.adapters[2].availability.status, 'unavailable');
  assert.equal(payload.lastIngestion?.id, report.ingestion.id);
  assert.equal(payload.sources.length, 8, 'containers, represented files, unsupported files, and failures stay visible');

  const sourcesDirectory = payload.sources.find((source) => source.observation.relPath === 'sources')!;
  const ideasDirectory = payload.sources.find((source) => source.observation.relPath === 'sources/ideas')!;
  const alpha = payload.sources.find((source) => source.observation.relPath === 'sources/ideas/alpha.md')!;
  const loose = payload.sources.find((source) => source.observation.relPath === 'loose.md')!;
  assert.equal(sourcesDirectory.parentSourceId, null);
  assert.equal(sourcesDirectory.name, 'sources');
  assert.equal(ideasDirectory.parentSourceId, sourcesDirectory.source.id);
  assert.equal(alpha.parentSourceId, ideasDirectory.source.id);
  assert.equal(loose.parentSourceId, null, 'an explicit root file mounts at the virtual workspace root');

  const textSource = payload.sources.find((source) => source.observation.relPath === 'sources/note.txt')!;
  assert.equal(textSource.lastResult?.status, 'imported');
  assert.equal(textSource.presence, 'present');
  assert.equal(textSource.representation?.adapter.id, 'headspace.text.native');
  const textBinding = payload.bindings.find((binding) => binding.relPath === 'sources/note.txt')!;
  assert.equal(textBinding.docChunkId, textSource.representation?.rootChunkId);
  assert.equal(textBinding.sourceId, textSource.source.id);
  assert.equal(textBinding.mediaType, 'text/plain');

  const unsupported = payload.sources.find((source) => source.observation.relPath === 'sources/image.bin')!;
  assert.equal(unsupported.lastResult?.status, 'unsupported');
  assert.equal(unsupported.representation, null);
  assert.equal(unsupported.parentSourceId, sourcesDirectory.source.id);
  const failed = payload.sources.find((source) => source.observation.relPath === 'sources/invalid.txt')!;
  assert.equal(failed.lastResult?.status, 'failed');
  assert.equal(failed.parentSourceId, sourcesDirectory.source.id);
  assert.equal(failed.lastResult?.adapter?.id, 'headspace.text.native');
  assert.match(failed.lastResult?.diagnostics[0].message ?? '', /encoded data|UTF-8|utf-8/i);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(payload)), 'the API payload is JSON-safe');

  const workspaceId = payload.workspace.id;
  ws.close();
  ws = await openWorkspace(root);
  const restarted = workspacePayload(root, ws);
  assert.equal(restarted.workspace.id, workspaceId);
  assert.equal(
    restarted.sources.find((source) => source.source.id === ideasDirectory.source.id)?.parentSourceId,
    sourcesDirectory.source.id,
  );
  assert.equal(restarted.sources.find((source) => source.source.id === failed.source.id)?.lastResult?.status, 'failed');

  rmSync(join(root, 'sources', 'note.txt'));
  await syncWorkspace(ws, { contentDirs: ['sources'], contentFiles: ['loose.md'] });
  const afterRemoval = workspacePayload(root, ws);
  const missingText = afterRemoval.sources.find((source) => source.source.id === textSource.source.id)!;
  assert.equal(missingText.presence, 'missing');
  assert.equal(missingText.lastResult, null);
  assert.equal(
    missingText.representation?.rootChunkId,
    textSource.representation?.rootChunkId,
    'a missing external source retains its recoverable substrate representation',
  );
  ws.close();
  ws = await openWorkspace(root);
  assert.equal(
    workspacePayload(root, ws).sources.find((source) => source.source.id === textSource.source.id)?.presence,
    'missing',
    'missing presence survives restart',
  );

  console.log('ingestion API contract OK — sources, capabilities, results, and bindings are observable');
} finally {
  ws?.close();
  rmSync(root, { recursive: true, force: true });
}
