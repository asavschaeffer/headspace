import { emptyState, renderChunk, materialize, childOccurrences, openProposals } from '../src/kernel/state';
import {
  createChunk,
  revise,
  transclude,
  promoteExtract,
  acceptProposal,
  scanWatchedSources,
  moveOccurrence,
  type TxCtx,
} from '../src/kernel/tx';
import { generateProposal } from '../src/kernel/select';
import { decomposeText, mdBlockSpans } from '../src/kernel/decompose';
import { keyBetween } from '../src/kernel/fractional';
import type { Commit } from '../src/kernel/types';
import { MEDIA_COMPOSITE } from '../src/kernel/types';
import assert from 'node:assert';

const log: Commit[] = [];
const ctx: TxCtx = { state: emptyState(), actorId: 'human:asa', onCommit: (c) => log.push(c) };

// doc with two blocks
const doc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
const b1 = await createChunk(ctx, { text: 'the lake is still', containerId: doc.chunkId });
const b2 = await createChunk(ctx, { text: 'the sky above it burns', containerId: doc.chunkId });
assert.equal(renderChunk(ctx.state, doc.chunkId), 'the lake is still\n\nthe sky above it burns');

// revise keeps identity, new revision
const before = ctx.state.chunks.get(b1.chunkId)!.currentRevisionId;
await revise(ctx, { chunkId: b1.chunkId, text: 'the lake is stormy' });
const after = ctx.state.chunks.get(b1.chunkId)!.currentRevisionId;
assert.notEqual(before, after);
assert.equal(ctx.state.revisions.get(before)!.chunkId, b1.chunkId);
assert.equal(renderChunk(ctx.state, doc.chunkId), 'the lake is stormy\n\nthe sky above it burns');

// reorder is not a content edit
const occs = childOccurrences(ctx.state, doc.chunkId);
moveOccurrence(ctx, { occurrenceId: occs[1].id, at: 'start' });
assert.equal(renderChunk(ctx.state, doc.chunkId), 'the sky above it burns\n\nthe lake is stormy');
assert.equal(ctx.state.chunks.get(doc.chunkId)!.currentRevisionId, doc.revisionId, 'doc revision unchanged by move');

// extraction: render-preserving
const rev = ctx.state.chunks.get(b2.chunkId)!.currentRevisionId;
const text = 'the sky above it burns';
const start = text.indexOf('above'), end = start + 'above it'.length;
const ex = await promoteExtract(ctx, { span: { revisionId: rev, method: 'raw@1', start, end } });
assert.equal(renderChunk(ctx.state, b2.chunkId), text, 'extract preserves rendered text');
assert.equal(ex.partChunkIds.length, 3);

// generation → proposal → accept
const gen = await generateProposal(ctx, { focusChunkId: doc.chunkId, instruction: 'continue' });
assert.equal(openProposals(ctx.state).length, 1);
const acc = await acceptProposal(ctx, { proposalId: gen.proposalId });
assert.ok(acc.applied);
assert.equal(childOccurrences(ctx.state, doc.chunkId).length, 3);
const genChunk = acc.createdChunkIds[0];
assert.equal(ctx.state.revisions.get(ctx.state.chunks.get(genChunk)!.currentRevisionId)!.createdBy, 'agent:stub');

// watched transclusion: revise source → proposal raised → accept repins
const doc2 = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
const tr = transclude(ctx, { containerId: doc2.chunkId, sourceChunkId: b1.chunkId });
assert.equal(renderChunk(ctx.state, doc2.chunkId), 'the lake is stormy');
await revise(ctx, { chunkId: b1.chunkId, text: 'the lake is frozen' });
assert.equal(renderChunk(ctx.state, doc2.chunkId), 'the lake is stormy', 'pinned render is stable');
const raised = scanWatchedSources(ctx);
assert.equal(raised.length, 1);
const upd = await acceptProposal(ctx, { proposalId: raised[0] });
assert.ok(upd.applied);
assert.equal(renderChunk(ctx.state, doc2.chunkId), 'the lake is frozen');

// stale generation proposal → superseded: the target advanced past the basis,
// so the context the generator saw no longer exists (wiki/operations.md)
const gen2 = await generateProposal(ctx, { focusChunkId: b1.chunkId, instruction: 'noop' });
await revise(ctx, { chunkId: b1.chunkId, text: 'the lake is gone' });
const acc2 = await acceptProposal(ctx, { proposalId: gen2.proposalId });
assert.ok(!acc2.applied, 'generation proposals go stale when their target advances');
assert.equal(ctx.state.proposals.get(gen2.proposalId)!.status, 'superseded');

// forged blob: same hash, different content — rejected, history not rewritten
{
  const victim = ctx.state.revisions.get(ctx.state.chunks.get(b1.chunkId)!.currentRevisionId)!;
  const originalText = ctx.state.blobs.get(victim.blobHash)!.text;
  let rejected = false;
  try {
    const { applyCommit } = await import('../src/kernel/state');
    applyCommit(ctx.state, {
      id: 'cmt_forged',
      parentIds: ctx.state.head ? [ctx.state.head] : [],
      at: new Date().toISOString(),
      actorId: 'human:mallory',
      operation: {
        id: 'op_forged',
        kind: 'create',
        actorId: 'human:mallory',
        at: new Date().toISOString(),
        inputRevisionIds: [],
        outputRevisionIds: [],
      },
      facts: { blobs: [{ hash: victim.blobHash, mediaType: victim.mediaType, text: 'tampered' }] },
    });
  } catch {
    rejected = true;
  }
  assert.ok(rejected, 'forged blob commit must be rejected');
  assert.equal(ctx.state.blobs.get(victim.blobHash)!.text, originalText, 'immutable payload unchanged');
}

// atomicity: a commit with valid facts plus one invalid tombstone leaves nothing behind
{
  const { applyCommit } = await import('../src/kernel/state');
  const chunksBefore = ctx.state.chunks.size;
  const headBefore = ctx.state.head;
  let rejected = false;
  try {
    applyCommit(ctx.state, {
      id: 'cmt_partial',
      parentIds: headBefore ? [headBefore] : [],
      at: new Date().toISOString(),
      actorId: 'human:asa',
      operation: {
        id: 'op_partial',
        kind: 'tombstone',
        actorId: 'human:asa',
        at: new Date().toISOString(),
        inputRevisionIds: [],
        outputRevisionIds: [],
      },
      facts: {
        blobs: [{ hash: 'deadbeef', mediaType: 'text/plain', text: 'orphan' }],
        tombstone: ['ch_nonexistent'],
      },
    });
  } catch {
    rejected = true;
  }
  assert.ok(rejected, 'invalid tombstone must be rejected');
  assert.equal(ctx.state.chunks.size, chunksBefore, 'no partial facts folded');
  assert.equal(ctx.state.head, headBefore, 'head untouched');
  assert.ok(!ctx.state.blobs.has('deadbeef'), 'no partial blob folded');
}

// log replay equivalence
const rebuilt = materialize(log);
assert.equal(rebuilt.chunks.size, ctx.state.chunks.size);
assert.equal(rebuilt.occurrences.size, ctx.state.occurrences.size);
assert.equal(renderChunk(rebuilt, doc2.chunkId), 'the lake is frozen');

// cycle rejection
let threw = false;
try {
  transclude(ctx, { containerId: b1.chunkId, sourceChunkId: doc.chunkId });
  const occsDoc = childOccurrences(ctx.state, b1.chunkId);
  void occsDoc;
} catch {
  threw = true;
}
assert.ok(threw, 'containment cycle rejected');

// fractional keys
let a = keyBetween(null, null);
let bnd = keyBetween(a, null);
for (let i = 0; i < 200; i++) {
  const m = keyBetween(a, bnd);
  assert.ok(a < m && m < bnd, `ordering broken at ${i}: ${a} ${m} ${bnd}`);
  bnd = m;
}

// markdown block spans
const md = '# Title\n\npara one\nline two\n\n```js\ncode\n\nstill code\n```\n\n- item a\n- item b\n';
const spans = mdBlockSpans(md);
assert.equal(spans.length, 4, `expected 4 blocks, got ${JSON.stringify(spans.map((s) => md.slice(s.start, s.end)))}`);
assert.equal(md.slice(spans[0].start, spans[0].end), '# Title');
assert.equal(md.slice(spans[2].start, spans[2].end), '```js\ncode\n\nstill code\n```');
assert.equal(decomposeText('One sentence. Two sentence! Three?', 'sent/icu@1').length, 3);

console.log('smoke OK —', log.length, 'commits,', ctx.state.chunks.size, 'chunks');
