import assert from 'node:assert';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { childOccurrences, emptyState } from '../src/kernel/state';
import { acceptProposal, revise, type TxCtx } from '../src/kernel/tx';
import {
  importMarkdownFile,
  projectMarkdown,
  reconcileMarkdownFile,
  sidecarPath,
  writeProjection,
  type MarkdownSidecar,
} from '../src/host/markdown';
import { levenshtein, similarity } from '../src/host/similarity';

// similarity basics
assert.equal(levenshtein('kitten', 'sitting'), 3);
assert.equal(similarity('', ''), 1);
assert.equal(similarity('abc', 'abc'), 1);
assert.equal(similarity('abc', 'xyz'), 0);
assert.ok(similarity('para one edited\nline two', 'para one\nline two') >= 0.5);

const root = mkdtempSync(join(tmpdir(), 'substrate-md-'));
try {
  const state = emptyState();
  const dctx: TxCtx = { state, actorId: 'driver:fs' };
  const hctx: TxCtx = { state, actorId: 'human:asa' };
  const rel = 'notes/demo.md';
  const readSc = (): MarkdownSidecar => JSON.parse(readFileSync(sidecarPath(root, rel), 'utf8'));

  const A = '# Title';
  const B = 'para one\nline two';
  const C = '```js\ncode\n```';
  const D = '- item a\n- item b';
  const mk = (bs: string[]) => `${bs.join('\n\n')}\n`;

  // canonical round-trip: import then project is byte-identical
  const md1 = mk([A, B, C, D]);
  const { docChunkId, blockChunkIds } = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md1 });
  assert.equal(blockChunkIds.length, 4);
  assert.equal(projectMarkdown(state, docChunkId), md1, 'canonical round-trip is byte-identical');
  const sc1 = readSc();
  assert.equal(sc1.docChunkId, docChunkId);
  assert.deepEqual(sc1.blocks.map((b) => b.chunkId), blockChunkIds);
  assert.equal(sc1.lastImportedFileHash, sc1.lastProjectedFileHash);
  assert.ok(!md1.includes(docChunkId), 'no ids in the md');

  // unchanged file → noop
  assert.equal((await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md1 })).action, 'noop');

  // external edit fast path: identity preserved, only the changed block revised
  const B2 = 'para one edited\nline two';
  const md2 = mk([A, B2, C, D]);
  const revBefore = blockChunkIds.map((id) => state.chunks.get(id)!.currentRevisionId);
  const r2 = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md2 });
  assert.equal(r2.action, 'fast-forward');
  assert.equal(r2.proposalId, undefined);
  assert.deepEqual(readSc().blocks.map((b) => b.chunkId), blockChunkIds, 'chunk identity preserved across reconcile');
  blockChunkIds.forEach((id, i) => {
    const now = state.chunks.get(id)!.currentRevisionId;
    if (i === 1) {
      assert.notEqual(now, revBefore[i], 'edited block revised');
      assert.equal(state.revisions.get(now)!.createdBy, 'driver:fs');
    } else {
      assert.equal(now, revBefore[i], 'untouched block not revised');
    }
  });
  assert.equal(projectMarkdown(state, docChunkId), md2);

  // external reorder fast path: moves only, no new chunks
  const md2b = mk([A, B2, D, C]);
  const chunksBefore = state.chunks.size;
  const r2b = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md2b });
  assert.equal(r2b.action, 'fast-forward');
  assert.equal(state.chunks.size, chunksBefore, 'reorder mints no chunks');
  assert.equal(projectMarkdown(state, docChunkId), md2b);

  // vanished block: sever proposal, not deletion
  const md3 = mk([A, B2, D]);
  const r3 = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md3 });
  assert.equal(r3.action, 'fast-forward');
  assert.ok(r3.proposalId, 'vanished block raises a proposal');
  assert.equal(childOccurrences(state, docChunkId).length, 4, 'occurrence survives until accept');
  const p3 = state.proposals.get(r3.proposalId!)!;
  assert.equal(p3.kind, 'reconciliation');
  assert.equal(p3.status, 'open');
  assert.deepEqual(p3.payload.map((c) => c.op), ['sever']);
  const acc3 = await acceptProposal(hctx, { proposalId: r3.proposalId! });
  assert.ok(acc3.applied);
  assert.equal(childOccurrences(state, docChunkId).length, 3);
  assert.equal(projectMarkdown(state, docChunkId), md3);

  // both sides changed: one reconciliation proposal, no direct mutation
  await revise(hctx, { chunkId: blockChunkIds[0], text: '# Renamed' });
  const B3 = 'para one edited twice\nline two';
  const md4 = mk([A, B3, D]); // file still carries the pre-rename title
  const scBytesBefore = readFileSync(sidecarPath(root, rel), 'utf8');
  const bRevBefore = state.chunks.get(blockChunkIds[1])!.currentRevisionId;
  const r4 = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: md4 });
  assert.equal(r4.action, 'proposal');
  assert.ok(r4.proposalId);
  assert.equal(state.chunks.get(blockChunkIds[1])!.currentRevisionId, bRevBefore, 'no direct revise while dirty');
  assert.equal(childOccurrences(state, docChunkId).length, 3, 'no direct placement while dirty');
  assert.equal(readFileSync(sidecarPath(root, rel), 'utf8'), scBytesBefore, 'sidecar untouched when proposing');
  const p4 = state.proposals.get(r4.proposalId!)!;
  assert.equal(p4.kind, 'reconciliation');
  assert.deepEqual(p4.payload.map((c) => c.op), ['revise']);
  const acc4 = await acceptProposal(hctx, { proposalId: r4.proposalId! });
  assert.ok(acc4.applied, 'reconciliation proposal applies on accept');
  assert.equal(projectMarkdown(state, docChunkId), mk(['# Renamed', B3, D]), 'both sides survive the merge');
  const bRevNow = state.chunks.get(blockChunkIds[1])!.currentRevisionId;
  assert.equal(state.revisions.get(bRevNow)!.createdBy, 'driver:fs', 'external text attributed to the driver');

  // projection writes the file and reconverges the sidecar
  await writeProjection(dctx, { workspaceRoot: root, relPath: rel });
  const onDisk = readFileSync(join(root, rel), 'utf8');
  assert.equal(onDisk, projectMarkdown(state, docChunkId));
  assert.equal((await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: onDisk })).action, 'noop');

  // path traversal rejected before anything is written
  await assert.rejects(importMarkdownFile(dctx, { workspaceRoot: root, relPath: '../escape.md', text: 'x\n' }));
  await assert.rejects(importMarkdownFile(dctx, { workspaceRoot: root, relPath: 'a/../../escape.md', text: 'x\n' }));
  await assert.rejects(importMarkdownFile(dctx, { workspaceRoot: root, relPath: resolve(root, '..', 'abs.md'), text: 'x\n' }));
  assert.throws(() => sidecarPath(root, '..'));
  assert.throws(() => sidecarPath(root, '.'));

  console.log('markdown driver OK —', state.chunks.size, 'chunks,', state.proposals.size, 'proposals');
} finally {
  rmSync(root, { recursive: true, force: true });
}
