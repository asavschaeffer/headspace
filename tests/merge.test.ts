import assert from 'node:assert';
import { commonAncestor, diff3, diffLines, proposeMerge } from '../src/kernel/merge';
import { currentRevision, emptyState, revisionText } from '../src/kernel/state';
import { acceptProposal, createChunk, revise, type TxCtx } from '../src/kernel/tx';

// ── diffLines ────────────────────────────────────────────────────────────────

assert.deepStrictEqual(diffLines('same\ntext', 'same\ntext'), []);
assert.deepStrictEqual(diffLines('a\nb\nc', 'a\nB\nc'), [{ aStart: 1, aLines: ['b'], bStart: 1, bLines: ['B'] }]);
assert.deepStrictEqual(diffLines('', 'x'), [{ aStart: 0, aLines: [], bStart: 0, bLines: ['x'] }]);
assert.deepStrictEqual(diffLines('x\ny', 'y'), [{ aStart: 0, aLines: ['x'], bStart: 0, bLines: [] }]);

// ── diff3: edits to different lines combine cleanly ──────────────────────────

{
  const base = 'one\ntwo\nthree\nfour\nfive';
  const r = diff3(base, 'ONE\ntwo\nthree\nfour\nfive', 'one\ntwo\nthree\nfour\nFIVE');
  assert.ok(r.clean);
  assert.strictEqual(r.merged, 'ONE\ntwo\nthree\nfour\nFIVE');
  assert.strictEqual(r.conflicts.length, 0);
}

// append on one side, edit far away on the other — still clean
{
  const r = diff3('a\nb\nc\nd', 'a\nb\nc\nd\ne', 'A\nb\nc\nd');
  assert.ok(r.clean);
  assert.strictEqual(r.merged, 'A\nb\nc\nd\ne');
}

// both sides making the identical change is not a conflict
{
  const r = diff3('x\ny', 'x\nz', 'x\nz');
  assert.ok(r.clean);
  assert.strictEqual(r.merged, 'x\nz');
}

// ── diff3: same-line conflict carries both sides verbatim ────────────────────

{
  const r = diff3('alpha\nbeta\ngamma', 'alpha\nours beta\ngamma', 'alpha\ntheirs beta\ngamma');
  assert.ok(!r.clean);
  assert.strictEqual(r.conflicts.length, 1);
  assert.deepStrictEqual(r.conflicts[0], {
    baseLines: ['beta'],
    oursLines: ['ours beta'],
    theirsLines: ['theirs beta'],
  });
  assert.strictEqual(r.merged, 'alpha\n<<<<<<< ours\nours beta\n=======\ntheirs beta\n>>>>>>> theirs\ngamma');
}

// ── DAG: fork, common ancestor, merge proposal, two-parent accept ────────────

const ctx: TxCtx = { state: emptyState(), actorId: 'human:asa' };

const doc = await createChunk(ctx, { text: 'one\ntwo\nthree\nfour\nfive' });
const baseRev = doc.revisionId;
const oursRev = (await revise(ctx, { chunkId: doc.chunkId, text: 'ONE\ntwo\nthree\nfour\nfive' })).revisionId;
// second head forked from base, not from ours
const theirsRev = (
  await revise(ctx, { chunkId: doc.chunkId, text: 'one\ntwo\nthree\nfour\nFIVE', mergeParentRevisionIds: [baseRev] })
).revisionId;

assert.strictEqual(commonAncestor(ctx.state, oursRev, theirsRev), baseRev);
assert.strictEqual(commonAncestor(ctx.state, oursRev, baseRev), baseRev, 'one head ancestor of the other');
assert.strictEqual(commonAncestor(ctx.state, oursRev, oursRev), oursRev);

const m = proposeMerge(ctx, { chunkId: doc.chunkId, oursRevisionId: oursRev, theirsRevisionId: theirsRev });
assert.ok(m.clean);
const p = ctx.state.proposals.get(m.proposalId)!;
assert.strictEqual(p.kind, 'merge');
assert.strictEqual(p.status, 'open', 'a clean merge is still a proposal');
assert.deepStrictEqual(p.basisRevisionIds, [oursRev, theirsRev]);
assert.strictEqual(p.note, 'clean merge');

const acc = await acceptProposal(ctx, { proposalId: m.proposalId });
assert.ok(acc.applied);
const head = currentRevision(ctx.state, doc.chunkId);
assert.deepStrictEqual(head.parentRevisionIds, [oursRev, theirsRev], 'merge revision names both parents');
assert.strictEqual(revisionText(ctx.state, head.id), 'ONE\ntwo\nthree\nfour\nFIVE');

// ── unrelated histories: base '', everything conflicts, nothing lost ─────────

const stray = await createChunk(ctx, { text: 'left line' });
// explicit empty mergeParentRevisionIds mints a rootless head: unrelated history
const rootless = (await revise(ctx, { chunkId: stray.chunkId, text: 'right line', mergeParentRevisionIds: [] })).revisionId;
assert.strictEqual(commonAncestor(ctx.state, stray.revisionId, rootless), null);

const m2 = proposeMerge(ctx, { chunkId: stray.chunkId, oursRevisionId: stray.revisionId, theirsRevisionId: rootless });
assert.ok(!m2.clean);
const p2 = ctx.state.proposals.get(m2.proposalId)!;
assert.match(p2.note!, /1 conflict/);
const change = p2.payload[0];
assert.ok(change.op === 'revise');
assert.deepStrictEqual(change.mergeParentRevisionIds, [stray.revisionId, rootless]);
assert.ok(change.text.includes('left line') && change.text.includes('right line'), 'both sides survive');
assert.strictEqual(change.text, '<<<<<<< ours\nleft line\n=======\nright line\n>>>>>>> theirs');

console.log('merge OK — diff3, ancestry, and two-parent merge proposals');
