import assert from 'node:assert';
import { blobHashOf } from '../src/kernel/hash';
import { emptyState } from '../src/kernel/state';
import { copyChunk, createChunk, redactRevision, revise, tombstoneChunk, type TxCtx } from '../src/kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_TEXT } from '../src/kernel/types';
import {
  buildIndexes,
  duplicatesOf,
  echoesOf,
  firstSeen,
  provenanceKind,
  searchChunks,
} from '../src/index/indexes';

// Deterministic, strictly increasing clock so firstSeen ordering is testable.
let tick = 0;
const now = () => new Date(Date.UTC(2026, 0, 1, 0, 0, tick++)).toISOString();
const ctx: TxCtx = { state: emptyState(), actorId: 'human:asa', now };
const state = ctx.state;

const doc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
const a = await createChunk(ctx, { text: 'The lake is still tonight. Go now.', containerId: doc.chunkId });
const b = await createChunk(ctx, {
  text: 'The lake is still tonight. The sky above the water burns. Go now.',
  containerId: doc.chunkId,
});
const c = await createChunk(ctx, { text: 'lake lake lake', containerId: doc.chunkId });

let ix = buildIndexes(state);

// ── search: finds, prefix-matches, ranks by match count ──────────────────────
let hits = searchChunks(state, ix, 'lake');
assert.deepEqual(new Set(hits), new Set([a.chunkId, b.chunkId, c.chunkId]));
assert.equal(hits[0], c.chunkId, 'three postings outrank one');
assert.deepEqual(searchChunks(state, ix, 'lak'), hits, 'query tokens prefix-match indexed tokens');
assert.deepEqual(searchChunks(state, ix, 'lake burns'), [b.chunkId], 'every token must match the same chunk');
assert.deepEqual(searchChunks(state, ix, 'stillness'), [], 'indexed token is not a prefix of the query token');
assert.deepEqual(searchChunks(state, ix, 'join'), [], 'composite blobs are config, not indexed content');
assert.deepEqual(searchChunks(state, ix, '...'), [], 'no word tokens, no result');

// NFC normalization: composed and decomposed spellings meet in one token.
const nfc = await createChunk(ctx, { text: 'café noir table' }); // decomposed é
ix = buildIndexes(state);
assert.deepEqual(searchChunks(state, ix, 'café'), [nfc.chunkId]);

// ── echoes: identical sentence across chunks; short sentences ignored ────────
const echoes = echoesOf(state, ix, a.chunkId);
assert.equal(echoes.length, 1, '"Go now." (2 words) must not echo');
assert.equal(echoes[0].text, 'The lake is still tonight.');
assert.deepEqual(echoes[0].others, [b.chunkId]);
assert.deepEqual(echoesOf(state, ix, c.chunkId), [], 'unshared sentences are not echoes');

// ── interning: history included; duplicates only over current revisions ──────
const d = await createChunk(ctx, { text: 'ephemeral first draft' });
const oldHash = await blobHashOf(MEDIA_TEXT, 'ephemeral first draft');
await revise(ctx, { chunkId: d.chunkId, text: 'polished second draft' });
ix = buildIndexes(state);
assert.deepEqual(ix.interning.get(oldHash), [d.revisionId], 'superseded revision stays interned');
assert.deepEqual(duplicatesOf(state, ix, oldHash), [], 'not any chunk\'s current content');
assert.deepEqual(searchChunks(state, ix, 'ephemeral'), [], 'term index covers current revisions only');
assert.deepEqual(searchChunks(state, ix, 'polished'), [d.chunkId]);

const dup = copyChunk(ctx, { sourceChunkId: c.chunkId });
const lakeHash = await blobHashOf(MEDIA_TEXT, 'lake lake lake');
ix = buildIndexes(state);
assert.deepEqual(new Set(duplicatesOf(state, ix, lakeHash)), new Set([c.chunkId, dup.chunkId]));
assert.equal(ix.interning.get(lakeHash)!.length, 2);

// ── firstSeen reflows on redaction (the deletion.md rule) ────────────────────
const mira = await createChunk(ctx, { text: 'the lake is still' });
const asa2 = await createChunk(ctx, { text: 'the lake is still' });
const phraseHash = await blobHashOf(MEDIA_TEXT, 'the lake is still');
ix = buildIndexes(state);
assert.equal(firstSeen(state, ix, phraseHash)!.id, mira.revisionId, 'earliest visible by createdAt');

redactRevision(ctx, { revisionId: mira.revisionId });
assert.equal(firstSeen(state, ix, phraseHash)!.id, asa2.revisionId, 'reflow needs no rebuild');
assert.deepEqual(searchChunks(state, ix, 'still tonight'), [a.chunkId, b.chunkId].sort());
ix = buildIndexes(state);
assert.equal(firstSeen(state, ix, phraseHash)!.id, asa2.revisionId, 'reflow survives rebuild');
assert.deepEqual(duplicatesOf(state, ix, phraseHash), [asa2.chunkId], 'redacted content is not a duplicate');
assert.ok(!ix.interning.has(phraseHash) || !ix.interning.get(phraseHash)!.includes(mira.revisionId));

redactRevision(ctx, { revisionId: asa2.revisionId });
ix = buildIndexes(state);
assert.equal(firstSeen(state, ix, phraseHash), null, 'no visible bearer left');

// ── tombstone: content stops surfacing everywhere ────────────────────────────
const ghost = await createChunk(ctx, { text: 'The lake is still tonight. Phantom syllable quandary rests.' });
ix = buildIndexes(state);
assert.deepEqual(searchChunks(state, ix, 'phantom'), [ghost.chunkId]);
assert.ok(echoesOf(state, ix, a.chunkId)[0].others.includes(ghost.chunkId));

tombstoneChunk(ctx, { chunkId: ghost.chunkId });
assert.deepEqual(searchChunks(state, ix, 'phantom'), [], 'stale index answers filter the tombstoned');
ix = buildIndexes(state);
assert.deepEqual(searchChunks(state, ix, 'phantom'), []);
assert.deepEqual(echoesOf(state, ix, a.chunkId)[0].others, [b.chunkId], 'tombstoned sharer drops out of echoes');
const ghostHash = await blobHashOf(MEDIA_TEXT, 'The lake is still tonight. Phantom syllable quandary rests.');
assert.deepEqual(duplicatesOf(state, ix, ghostHash), []);
assert.equal(firstSeen(state, ix, ghostHash), null);

// ── provenance ───────────────────────────────────────────────────────────────
assert.equal(provenanceKind(state, a.chunkId), 'human');
const agentCtx: TxCtx = { state, actorId: 'agent:stub', now };
const gen = await createChunk(agentCtx, { text: 'machine reverie output' });
assert.equal(provenanceKind(state, gen.chunkId), 'agent');

console.log('indexes OK —', ix.term.size, 'terms,', ix.interning.size, 'blobs,', ix.echo.size, 'echo keys');
