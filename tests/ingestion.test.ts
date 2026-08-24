import assert from 'node:assert';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { childOccurrences, currentRevision, renderChunk, revisionText } from '../src/kernel/state';
import { acceptProposal, revise } from '../src/kernel/tx';
import { MEDIA_MARKDOWN, MEDIA_TEXT } from '../src/kernel/types';
import {
  ingestWorkspace,
  ingestionAdapterCapabilities,
  readIngestionCatalog,
  type IngestionItemResult,
} from '../src/host/ingestion';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const envelope = mkdtempSync(join(tmpdir(), 'headspace-ingestion-'));
const root = join(envelope, 'workspace');
const content = join(root, 'content');
const nested = join(content, 'nested');
mkdirSync(nested, { recursive: true });
writeFileSync(join(content, 'alpha.md'), '# Alpha\n\nMarkdown body.\n');
writeFileSync(join(nested, 'note.txt'), 'plain text source');
writeFileSync(join(content, 'manual.pdf'), Buffer.from('%PDF-not-converted'));
writeFileSync(join(content, 'invalid.txt'), Buffer.from([0xc3, 0x28]));

const outside = join(envelope, 'outside.txt');
writeFileSync(outside, 'must not be ingested');
let linked = false;
try {
  symlinkSync(outside, join(content, 'linked.txt'), 'file');
  linked = true;
} catch (e) {
  const unavailable = new Set(['EACCES', 'EINVAL', 'ENOSYS', 'ENOTSUP', 'EPERM', 'UNKNOWN']);
  if (!unavailable.has((e as NodeJS.ErrnoException).code ?? '')) throw e;
}

const byPath = (items: IngestionItemResult[], relPath: string): IngestionItemResult => {
  const item = items.find((candidate) => candidate.observation.relPath === relPath);
  assert.ok(item, `missing ingestion result for ${relPath}`);
  return item;
};

let ws: WorkspaceStore | null = null;
try {
  ws = await openWorkspace(root);
  const capabilities = ingestionAdapterCapabilities();
  assert.deepEqual(
    capabilities.map((capability) => capability.id),
    ['headspace.markdown.native', 'headspace.text.native', 'headspace.pdf-to-markdown.http'],
  );
  assert.equal(capabilities[0].outputs[0].writeback, 'round-trip');
  assert.equal(capabilities[1].outputs[0].writeback, 'none', 'capabilities do not claim unimplemented text projection');
  assert.equal(capabilities[2].availability.status, 'unavailable');
  assert.equal(
    capabilities[2].availability.status === 'unavailable' && capabilities[2].availability.diagnostic.code,
    'adapter.pdf-converter-unconfigured',
  );

  const first = await ingestWorkspace(ws, {
    contentDirs: ['content'],
    contentFiles: ['content/nested/note.txt'],
  });
  assert.equal(
    first.items.filter((item) => item.observation.relPath === 'content/nested/note.txt').length,
    1,
    'overlapping configured inputs observe one canonical source',
  );
  assert.equal(byPath(first.items, 'content').observation.kind, 'directory');
  assert.equal(byPath(first.items, 'content/nested').observation.kind, 'directory');

  const markdown = byPath(first.items, 'content/alpha.md');
  assert.equal(markdown.status, 'imported');
  assert.equal(markdown.adapter?.id, 'headspace.markdown.native');
  assert.ok(markdown.representation);
  assert.equal(markdown.representation!.mediaType, MEDIA_MARKDOWN);

  const text = byPath(first.items, 'content/nested/note.txt');
  assert.equal(text.status, 'imported');
  assert.equal(text.adapter?.id, 'headspace.text.native');
  assert.ok(text.representation);
  assert.equal(text.representation!.contentChunkIds.length, 1);
  assert.equal(text.observation.fingerprint.algorithm, 'sha256');
  assert.equal(text.observation.fingerprint.basis, 'file-bytes');
  assert.equal(text.observation.fingerprint.value.length, 64);
  assert.equal(renderChunk(ws.state, text.representation!.rootChunkId), 'plain text source');
  const textLeafId = text.representation!.contentChunkIds[0];
  assert.equal(currentRevision(ws.state, textLeafId).mediaType, MEDIA_TEXT);
  assert.equal(currentRevision(ws.state, textLeafId).createdBy, 'adapter:headspace.text.native@1.0.0');
  assert.equal(text.representation!.operationIds.length, 1);
  const textImportOperation = ws.state.operations.get(text.representation!.operationIds[0])!;
  assert.deepEqual(textImportOperation.outputRevisionIds, text.representation!.outputRevisionIds);
  assert.equal((textImportOperation.params as Record<string, unknown>).sourceId, text.observation.sourceId);
  assert.equal((textImportOperation.params as Record<string, unknown>).observationId, text.observation.id);

  const unsupported = byPath(first.items, 'content/manual.pdf');
  assert.equal(unsupported.status, 'unsupported');
  assert.equal(unsupported.observation.mediaType, 'application/pdf');
  assert.equal(unsupported.adapter?.id, 'headspace.pdf-to-markdown.http');
  assert.equal(unsupported.diagnostics[0].code, 'adapter.pdf-converter-unconfigured');

  const invalid = byPath(first.items, 'content/invalid.txt');
  assert.equal(invalid.status, 'failed');
  assert.equal(invalid.adapter?.id, 'headspace.text.native');
  assert.equal(invalid.diagnostics[0].code, 'adapter.ingest-failed');

  if (linked) {
    const link = byPath(first.items, 'content/linked.txt');
    assert.equal(link.observation.kind, 'symlink');
    assert.equal(link.observation.symlink.status, 'unfollowed-outside-root');
    assert.equal(link.status, 'unsupported');
  }

  const firstCatalog = readIngestionCatalog(root)!;
  assert.equal(firstCatalog.sources.length, first.items.length, 'every unique observation has a durable source record');
  assert.equal(firstCatalog.pendingMaterializations.length, 0, 'completed imports leave no write-ahead intents');
  const firstTextSource = firstCatalog.sources.find((source) => source.currentRelPath === 'content/nested/note.txt')!;
  assert.equal(firstTextSource.identityKey, text.observation.identityKey);
  const firstTextRoot = text.representation!.rootChunkId;
  const firstMarkdownRoot = markdown.representation!.rootChunkId;
  const firstWorkspaceId = firstCatalog.workspaceId;

  ws.close();
  ws = await openWorkspace(root);
  const second = await ingestWorkspace(ws, { contentDirs: ['content'] });
  const secondText = byPath(second.items, 'content/nested/note.txt');
  const secondMarkdown = byPath(second.items, 'content/alpha.md');
  assert.equal(secondText.status, 'unchanged');
  assert.equal(secondMarkdown.status, 'unchanged');
  assert.equal(secondText.observation.sourceId, firstTextSource.id, 'source identity survives process restart');
  assert.equal(secondText.representation?.rootChunkId, firstTextRoot, 'text chunk identity survives process restart');
  assert.equal(secondMarkdown.representation?.rootChunkId, firstMarkdownRoot, 'Markdown chunk identity survives process restart');
  assert.equal(readIngestionCatalog(root)?.workspaceId, firstWorkspaceId, 'workspace identity survives process restart');

  // A clean external text edit advances only the source-owned leaf.
  writeFileSync(join(nested, 'note.txt'), 'plain text changed outside');
  const third = await ingestWorkspace(ws, { contentDirs: ['content'] });
  const updatedText = byPath(third.items, 'content/nested/note.txt');
  assert.equal(updatedText.status, 'updated');
  assert.equal(updatedText.representation?.rootChunkId, firstTextRoot);
  assert.equal(renderChunk(ws.state, firstTextRoot), 'plain text changed outside');

  // Once Headspace has authored a newer revision, a further external edit is
  // an inert reconciliation proposal rather than a silent replacement.
  const latestRepresentation = updatedText.representation!;
  const leafId = latestRepresentation.contentChunkIds[0];
  await revise(ws.ctxFor('human:test'), { chunkId: leafId, text: 'internal authored text', mediaType: MEDIA_TEXT });
  const internalRevision = currentRevision(ws.state, leafId).id;
  writeFileSync(join(nested, 'note.txt'), 'another external version');
  const fourth = await ingestWorkspace(ws, { contentDirs: ['content'] });
  const proposedText = byPath(fourth.items, 'content/nested/note.txt');
  assert.equal(proposedText.status, 'proposal');
  assert.ok(proposedText.proposalId);
  assert.equal(currentRevision(ws.state, leafId).id, internalRevision);
  assert.equal(revisionText(ws.state, internalRevision), 'internal authored text');
  const proposal = ws.state.proposals.get(proposedText.proposalId!)!;
  assert.equal(proposal.kind, 'reconciliation');
  assert.deepEqual(proposal.payload.map((change) => change.op), ['revise']);

  ws.close();
  ws = await openWorkspace(root);
  const reopenedCatalog = readIngestionCatalog(root)!;
  assert.equal(reopenedCatalog.lastRun?.id, fourth.id, 'the observable run report survives restart');
  assert.equal(reopenedCatalog.sources.find((source) => source.currentRelPath === 'content/nested/note.txt')?.id, firstTextSource.id);
  assert.equal(renderChunk(ws.state, firstTextRoot), 'internal authored text');
  assert.equal(ws.state.proposals.get(proposedText.proposalId!)?.status, 'open');

  assert.ok((await acceptProposal(ws.ctxFor('human:test'), { proposalId: proposedText.proposalId! })).applied);
  const converged = await ingestWorkspace(ws, { contentDirs: ['content'] });
  const convergedText = byPath(converged.items, 'content/nested/note.txt');
  assert.equal(convergedText.status, 'updated', 'accepted external text converges its source provenance');
  assert.equal(renderChunk(ws.state, firstTextRoot), 'another external version');
  assert.equal(
    [...ws.state.proposals.values()].filter(
      (candidate) => candidate.status === 'open' && candidate.targetChunkIds.includes(leafId),
    ).length,
    0,
    'convergence does not raise a duplicate proposal',
  );

  console.log(
    `ingestion seam OK — ${first.items.length} observations, ${first.counts.imported} imported, ${first.counts.unsupported} unsupported`,
  );
} finally {
  ws?.close();
  rmSync(envelope, { recursive: true, force: true });
}
