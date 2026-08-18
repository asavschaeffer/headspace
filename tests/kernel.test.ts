// Adversarial kernel invariant tests. The wiki is the spec (kernel.md,
// operations.md, proposals.md, decomposition.md, deletion.md, janus.md,
// deep-fates.md); these tests hold the kernel to it through public exports only.
import { strict as assert } from 'node:assert';
import {
  applyCommit,
  currentRevision,
  emptyState,
  InvariantError,
  renderChunk,
  revisionText,
} from '../src/kernel/state';
import {
  acceptProposal,
  copyChunk,
  createChunk,
  placeOccurrence,
  promoteCopy,
  promoteExtract,
  propose,
  redactRevision,
  rejectProposal,
  revise,
  tombstoneChunk,
  transclude,
  type TxCtx,
} from '../src/kernel/tx';
import { reduce, select, type ContextItem } from '../src/kernel/select';
import { decomposeRevision, METHOD_BLOCKS, METHOD_SENTENCES, METHOD_WORDS } from '../src/kernel/decompose';
import { keyBetween } from '../src/kernel/fractional';
import { newCommitId, newOperationId } from '../src/kernel/ids';
import type { Commit, Facts } from '../src/kernel/types';
import { MEDIA_COMPOSITE, MEDIA_MARKDOWN } from '../src/kernel/types';

// Deterministic clock injected through TxCtx.now: every timestamp is assertable
// and strictly increasing (ms ticks roll over cleanly in Date.UTC).
let tick = 0;
const now = () => new Date(Date.UTC(2026, 0, 1, 0, 0, 0, tick++)).toISOString();
const ctx: TxCtx = { state: emptyState(), actorId: 'human:asa', now };

// A refused commit must be a non-event: same head, same count, no partial state.
const guard = (fn: () => unknown, re: RegExp) => {
  const head = ctx.state.head;
  const count = ctx.state.commitCount;
  assert.throws(fn, (e: unknown) => e instanceof InvariantError && re.test(String(e)));
  assert.equal(ctx.state.head, head, 'refused commit must not advance head');
  assert.equal(ctx.state.commitCount, count, 'refused commit must not append');
};

// ── revision immutability and current-pointer ownership (kernel.md) ──────────
const a = await createChunk(ctx, { text: 'alpha' });
assert.equal(ctx.state.revisions.get(a.revisionId)!.createdAt, '2026-01-01T00:00:00.000Z', 'injected clock stamps provenance');
assert.equal(a.commit.at, '2026-01-01T00:00:00.000Z');
const b = await createChunk(ctx, { text: 'beta' });

const mkCommit = (facts: Facts): Commit => ({
  id: newCommitId(),
  parentIds: ctx.state.head ? [ctx.state.head] : [],
  at: now(),
  actorId: ctx.actorId,
  operation: { id: newOperationId(), kind: 'revise', actorId: ctx.actorId, at: now(), inputRevisionIds: [], outputRevisionIds: [] },
  facts,
});
// Re-appending an existing revision id is refused even byte-identical: history is append-once.
guard(() => applyCommit(ctx.state, mkCommit({ revisions: [{ ...ctx.state.revisions.get(a.revisionId)! }] })), /immutable/);
// "A chunk's current revision must belong to that chunk."
guard(() => applyCommit(ctx.state, mkCommit({ setCurrent: [{ chunkId: a.chunkId, revisionId: b.revisionId }] })), /belongs to/);

// ── occurrences: unknown container, containment acyclicity (kernel.md) ───────
guard(() => placeOccurrence(ctx, { containerId: 'ch_ghost', chunkId: a.chunkId }), /unknown/);

const root = await createChunk(ctx, { text: 'root' });
const mid = await createChunk(ctx, { text: 'mid', containerId: root.chunkId });
const leaf = await createChunk(ctx, { text: 'leaf', containerId: mid.chunkId });
guard(() => placeOccurrence(ctx, { containerId: leaf.chunkId, chunkId: root.chunkId }), /cycle/);
// Transclusion is an occurrence, so it obeys the same acyclicity.
guard(() => transclude(ctx, { containerId: leaf.chunkId, sourceChunkId: root.chunkId }), /cycle/);
guard(() => placeOccurrence(ctx, { containerId: root.chunkId, chunkId: root.chunkId }), /cycle/);

// ── promotion, extract shape (decomposition.md) ──────────────────────────────
const para = await createChunk(ctx, { text: 'alpha beta gamma delta' });
await revise(ctx, { chunkId: para.chunkId, text: 'alpha beta gamma delta epsilon' });
// A span addresses one immutable revision; if the chunk moved on, extraction must refuse.
await assert.rejects(
  promoteExtract(ctx, { span: { revisionId: para.revisionId, method: 'raw@1', start: 0, end: 5 } }),
  /moved on/,
);
const flat = currentRevision(ctx.state, para.chunkId);
const full = revisionText(ctx.state, flat.id);
const start = full.indexOf('beta');
const end = start + 'beta gamma'.length;
const ex = await promoteExtract(ctx, { span: { revisionId: flat.id, method: 'raw@1', start, end } });
assert.equal(renderChunk(ctx.state, para.chunkId), full, 'extraction preserves the parent rendering');
assert.equal(renderChunk(ctx.state, ex.extractedChunkId), 'beta gamma');
const exDrv = [...ctx.state.derivations.values()].find((d) => d.childChunkId === ex.extractedChunkId);
assert.ok(exDrv && exDrv.via === 'extract');
assert.deepEqual(exDrv!.sourceSpan, { revisionId: flat.id, method: 'raw@1', start, end }, 'derivation carries the exact source span');
assert.equal(revisionText(ctx.state, flat.id), full, 'prior flat revision still resolves; history is sedimentary');
assert.equal(currentRevision(ctx.state, para.chunkId).mediaType, MEDIA_COMPOSITE);
assert.ok(currentRevision(ctx.state, para.chunkId).parentRevisionIds.includes(flat.id), 'composite descends from the flat revision');

// ── promotion, copy shape: only the asked-for promotion happens ──────────────
const src = await createChunk(ctx, { text: 'one two three' });
const headBefore = ctx.state.chunks.get(src.chunkId)!.currentRevisionId;
const chunksBefore = ctx.state.chunks.size;
const occsBefore = ctx.state.occurrences.size;
const pc = await promoteCopy(ctx, { span: { revisionId: headBefore, method: 'raw@1', start: 4, end: 7 } });
assert.equal(ctx.state.chunks.get(src.chunkId)!.currentRevisionId, headBefore, 'copy-shape promotion leaves the parent head untouched');
assert.equal(ctx.state.occurrences.size, occsBefore, 'no occurrence appears unasked');
assert.equal(ctx.state.chunks.size, chunksBefore + 1, 'exactly one durable object is minted');
assert.equal(renderChunk(ctx.state, pc.chunkId), 'two');
const pcDrv = [...ctx.state.derivations.values()].find((d) => d.childChunkId === pc.chunkId);
assert.deepEqual(pcDrv!.sourceSpan, { revisionId: headBefore, method: 'raw@1', start: 4, end: 7 });

// ── Janus: interning is not identity (janus.md, deep-fates.md) ───────────────
const orig = await createChunk(ctx, { text: 'the lake is still' });
const cp = copyChunk(ctx, { sourceChunkId: orig.chunkId });
assert.notEqual(cp.chunkId, orig.chunkId, 'copy mints a distinct chunk identity');
assert.equal(
  currentRevision(ctx.state, cp.chunkId).blobHash,
  currentRevision(ctx.state, orig.chunkId).blobHash,
  'identical payload shares the blob',
);
const cpDrv = [...ctx.state.derivations.values()].find((d) => d.childChunkId === cp.chunkId);
assert.ok(cpDrv && cpDrv.via === 'copy' && cpDrv.sourceRevisionId === orig.revisionId, 'copy records its ancestry');
await revise(ctx, { chunkId: cp.chunkId, text: 'the lake is stormy' });
assert.equal(renderChunk(ctx.state, orig.chunkId), 'the lake is still', 'editing the copy never reaches the source');

// ── proposal freshness: a stale suggested edit is refused (operations.md) ────
const target = await createChunk(ctx, { text: 'original wording' });
const sug = propose(ctx, {
  kind: 'suggested-edit',
  basisRevisionIds: [target.revisionId],
  targetChunkIds: [target.chunkId],
  payload: [{ op: 'revise', chunkId: target.chunkId, text: 'their suggestion' }],
  createdBy: 'human:mira',
});
await revise(ctx, { chunkId: target.chunkId, text: 'author moved on' });
const staleRes = await acceptProposal(ctx, { proposalId: sug.proposalId });
assert.equal(staleRes.applied, false, 'a payload only applies against the state it was computed from');
assert.match(staleRes.reason!, /moved on/);
assert.equal(ctx.state.proposals.get(sug.proposalId)!.status, 'superseded');
assert.equal(renderChunk(ctx.state, target.chunkId), 'author moved on', 'the refused payload touched nothing');
await assert.rejects(acceptProposal(ctx, { proposalId: sug.proposalId }), /superseded/, 'accepting a non-open proposal throws');

// ── reject is sedimentary (proposals.md) ─────────────────────────────────────
const sug2 = propose(ctx, {
  kind: 'suggested-edit',
  basisRevisionIds: [currentRevision(ctx.state, target.chunkId).id],
  targetChunkIds: [target.chunkId],
  payload: [{ op: 'revise', chunkId: target.chunkId, text: 'another suggestion' }],
  createdBy: 'human:mira',
});
rejectProposal(ctx, { proposalId: sug2.proposalId });
const kept = ctx.state.proposals.get(sug2.proposalId);
assert.ok(kept, 'a rejection is a recorded judgment, not an erasure');
assert.equal(kept!.status, 'rejected');
assert.equal(kept!.resolution!.by, 'human:asa');
assert.ok(kept!.resolution!.at > kept!.createdAt, 'resolution is clocked after creation');
await assert.rejects(acceptProposal(ctx, { proposalId: sug2.proposalId }), /rejected/);
assert.throws(() => rejectProposal(ctx, { proposalId: sug2.proposalId }), /rejected/);
assert.equal(renderChunk(ctx.state, target.chunkId), 'author moved on');

// ── accepted generation: two facts, not one (proposals.md) ───────────────────
const seed = await createChunk(ctx, { text: 'seed material' });
const gen = propose(ctx, {
  kind: 'generation',
  basisRevisionIds: [seed.revisionId],
  targetChunkIds: [seed.chunkId],
  payload: [
    {
      op: 'create',
      tempId: 'g',
      text: 'model words',
      mediaType: MEDIA_MARKDOWN,
      derivedFrom: { sourceRevisionId: seed.revisionId, via: 'generate' },
    },
  ],
  createdBy: 'agent:muse',
});
const acceptor: TxCtx = { state: ctx.state, actorId: 'human:mira', now };
const admitted = await acceptProposal(acceptor, { proposalId: gen.proposalId });
assert.ok(admitted.applied);
const genRev = currentRevision(ctx.state, admitted.createdChunkIds[0]);
assert.equal(genRev.createdBy, 'agent:muse', 'the model authored the content');
assert.equal(ctx.state.operations.get(genRev.operationId)!.actorId, 'human:mira', 'the acceptor admitted it');
assert.equal(ctx.state.proposals.get(gen.proposalId)!.resolution!.by, 'human:mira');
const genDrv = [...ctx.state.derivations.values()].find((d) => d.childChunkId === admitted.createdChunkIds[0]);
assert.ok(genDrv && genDrv.via === 'generate' && genDrv.sourceRevisionId === seed.revisionId);

// ── redaction hides content, keeps identity and shape (deletion.md) ──────────
const rdoc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
const secret = await createChunk(ctx, { text: 'the secret name', containerId: rdoc.chunkId });
await createChunk(ctx, { text: 'public part', containerId: rdoc.chunkId });
redactRevision(ctx, { revisionId: secret.revisionId });
assert.equal(renderChunk(ctx.state, secret.chunkId), '[redacted]');
assert.equal(renderChunk(ctx.state, rdoc.chunkId), '[redacted]\n\npublic part', 'redaction leaves a blank, not a hole');
assert.ok(ctx.state.chunks.get(secret.chunkId) && ctx.state.revisions.get(secret.revisionId), 'identity and shape remain');

// ── tombstoned chunks drop out of select() (deletion.md) ─────────────────────
const seldoc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
const keeper = await createChunk(ctx, { text: 'kept child', containerId: seldoc.chunkId });
const doomed = await createChunk(ctx, { text: 'doomed child', containerId: seldoc.chunkId });
assert.ok(select(ctx.state, seldoc.chunkId).some((i) => i.chunkId === doomed.chunkId));
tombstoneChunk(ctx, { chunkId: doomed.chunkId });
const items = select(ctx.state, seldoc.chunkId, [doomed.chunkId]);
assert.ok(!items.some((i) => i.chunkId === doomed.chunkId), 'tombstoned chunks are excluded, even as search hits');
assert.ok(items.some((i) => i.chunkId === keeper.chunkId));
assert.equal(revisionText(ctx.state, doomed.revisionId), 'doomed child', 'tombstone preserves identity/history');

// ── reduce: hard budget, spent in role priority (operations.md) ──────────────
const it = (role: ContextItem['role'], chunkId: string, len: number): ContextItem => ({
  chunkId,
  revisionId: `r-${chunkId}`,
  text: 'x'.repeat(len),
  role,
});
const ranked: ContextItem[] = [
  it('focus', 'f', 10),
  it('child', 'c1', 90),
  it('child', 'c2', 5000),
  it('parent', 'p', 0),
  it('sibling', 's', 50),
  it('search', 'q', 40),
];
const loose = reduce(ranked, 200);
assert.ok(loose.chars <= 200, 'budget is a hard bound');
assert.equal(loose.dropped, 1);
assert.ok(!loose.items.some((i) => i.chunkId === 'c2'), 'the over-budget item drops; the budget never stretches');
assert.ok(loose.items.some((i) => i.chunkId === 'p'), 'empty-text items ride free');
const tight = reduce(ranked, 100);
assert.ok(tight.chars <= 100);
assert.equal(tight.items[0].role, 'focus');
assert.deepEqual(
  tight.items.filter((i) => i.text.length > 0).map((i) => i.chunkId),
  ['f', 'c1'],
  'budget spends on higher-priority roles first',
);

// ── decomposition: spans slice back to their exact text (decomposition.md) ───
// Non-BMP 🌊 (surrogate pair) guards UTF-16 code-unit offsets against
// code-point counting; accents and dashes guard the trim math.
const tricky = 'Naïve 🌊 waves crest. Ces mots—liés—flottent!\nSecond thought? Done.';
const dch = await createChunk(ctx, { text: tricky });
for (const method of [METHOD_SENTENCES, METHOD_WORDS]) {
  const parts = decomposeRevision(ctx.state, currentRevision(ctx.state, dch.chunkId).id, method);
  assert.ok(parts.length > 0, `${method} found parts`);
  let prevEnd = 0;
  for (const p of parts) {
    assert.equal(p.text, tricky.slice(p.address.start, p.address.end), `${method} span slices back exactly`);
    assert.equal(p.text, p.text.trim(), `${method} offsets absorb the whitespace trim`);
    assert.ok(
      p.address.start >= prevEnd && p.address.start < p.address.end && p.address.end <= tricky.length,
      `${method} spans ordered and in range`,
    );
    prevEnd = p.address.end;
  }
}
const sentTexts = decomposeRevision(ctx.state, currentRevision(ctx.state, dch.chunkId).id, METHOD_SENTENCES).map((p) => p.text);
assert.equal(sentTexts.length, 4);
assert.equal(sentTexts[0], 'Naïve 🌊 waves crest.');
const wordTexts = decomposeRevision(ctx.state, currentRevision(ctx.state, dch.chunkId).id, METHOD_WORDS).map((p) => p.text);
for (const w of ['Naïve', 'waves', 'flottent', 'Done']) assert.ok(wordTexts.includes(w), `word "${w}" survives offset math`);

const md = '# Title\n\nfirst paragraph\ncontinues here\n\n```\ncode 🌊 fence\n\nstill code\n```\n';
const mch = await createChunk(ctx, { text: md, mediaType: MEDIA_MARKDOWN });
const blocks = decomposeRevision(ctx.state, currentRevision(ctx.state, mch.chunkId).id, METHOD_BLOCKS);
for (const p of blocks) assert.equal(p.text, md.slice(p.address.start, p.address.end));
assert.equal(blocks[0].text, '# Title');
assert.equal(blocks[2].text, '```\ncode 🌊 fence\n\nstill code\n```', 'fences stay whole, blank line included');

// ── fractional keys under 500 adversarial insertions ─────────────────────────
{
  const keys: string[] = [keyBetween(null, null)];
  for (let i = 0; i < 150; i++) keys.unshift(keyBetween(null, keys[0])); // hammer the front
  for (let i = 0; i < 150; i++) keys.push(keyBetween(keys[keys.length - 1], null)); // hammer the back
  let idx = Math.floor(keys.length / 2);
  // Repeatedly bisect the same pair: converge on the lower key, then the upper.
  for (let i = 0; i < 100; i++) keys.splice(idx + 1, 0, keyBetween(keys[idx], keys[idx + 1]));
  for (let i = 0; i < 100; i++) {
    keys.splice(idx + 1, 0, keyBetween(keys[idx], keys[idx + 1]));
    idx++;
  }
  assert.equal(keys.length, 501);
  for (let i = 1; i < keys.length; i++) {
    assert.ok(keys[i - 1] < keys[i], `key order broken at ${i}: "${keys[i - 1]}" !< "${keys[i]}"`);
  }
  // Alphabet-only, non-empty, never ending in the minimum digit — keeps every
  // key insertable-after forever.
  for (const k of keys) assert.match(k, /^[0-9A-Za-z]*[1-9A-Za-z]$/, `malformed key "${k}"`);
}

console.log('kernel invariants OK —', ctx.state.commitCount, 'commits,', ctx.state.chunks.size, 'chunks');
