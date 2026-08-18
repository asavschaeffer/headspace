// Filesystem workspace store: durability, replay, locking, crash artifacts.
// Each block owns a fresh tmp workspace; a deleted lock file simulates a crash
// (close() always snapshots, so crashing is the only way to exercise replay).
import assert from 'node:assert';
import { appendFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openWorkspace } from '../src/host/store-fs';
import { blobHashOf } from '../src/kernel/hash';
import { renderChunk } from '../src/kernel/state';
import { createChunk, revise } from '../src/kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_TEXT } from '../src/kernel/types';

const roots: string[] = [];
const freshRoot = () => {
  const d = mkdtempSync(join(tmpdir(), 'substrate-store-'));
  roots.push(d);
  return d;
};
const sub = (root: string, ...parts: string[]) => join(root, '.substrate', ...parts);
const crash = (root: string) => rmSync(sub(root, 'lock')); // died without close()

try {
  // ── write then reopen: log-only materialize, then snapshot path ────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const ctx = ws.ctxFor('human:asa');
    const doc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
    await createChunk(ctx, { text: 'the lake is still', containerId: doc.chunkId });
    const b2 = await createChunk(ctx, { text: 'the sky above it', containerId: doc.chunkId });
    await revise(ctx, { chunkId: b2.chunkId, text: 'the sky above it burns' });
    const rendered = renderChunk(ctx.state, doc.chunkId);
    const chunkCount = ctx.state.chunks.size;
    const occCount = ctx.state.occurrences.size;

    crash(root);
    assert.ok(!existsSync(sub(root, 'snapshot.json')), 'no snapshot was written before the crash');
    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.chunks.size, chunkCount);
    assert.equal(ws2.state.occurrences.size, occCount);
    assert.equal(renderChunk(ws2.state, doc.chunkId), rendered);
    ws2.close(); // writes the snapshot, releases the lock

    const ws3 = await openWorkspace(root);
    assert.equal(ws3.state.chunks.size, chunkCount);
    assert.equal(ws3.state.occurrences.size, occCount);
    assert.equal(renderChunk(ws3.state, doc.chunkId), rendered);
    ws3.close();
  }

  // ── stale snapshot + log tail replay ───────────────────────────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const ctx = ws.ctxFor('human:asa');
    const doc = await createChunk(ctx, { text: JSON.stringify({ join: ' / ' }), mediaType: MEDIA_COMPOSITE });
    await createChunk(ctx, { text: 'one', containerId: doc.chunkId });
    ws.saveSnapshot();
    await createChunk(ctx, { text: 'two', containerId: doc.chunkId });
    const b3 = await createChunk(ctx, { text: 'three', containerId: doc.chunkId });
    await revise(ctx, { chunkId: b3.chunkId, text: 'three, revised' });
    const rendered = renderChunk(ctx.state, doc.chunkId);

    const snap = JSON.parse(readFileSync(sub(root, 'snapshot.json'), 'utf8'));
    assert.equal(snap.coveredCommits, 2, 'snapshot covers only the first two commits');
    crash(root);
    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.commitCount, ctx.state.commitCount);
    assert.equal(ws2.state.chunks.size, ctx.state.chunks.size);
    assert.equal(ws2.state.occurrences.size, ctx.state.occurrences.size);
    assert.equal(renderChunk(ws2.state, doc.chunkId), rendered);
    assert.equal(renderChunk(ws2.state, doc.chunkId), 'one / two / three, revised');
    ws2.close();
  }

  // ── lock: conflict throws, force takes over, close releases ────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    await assert.rejects(() => openWorkspace(root), /locked by pid/);
    const forced = await openWorkspace(root, { force: true });
    forced.close();
    const again = await openWorkspace(root); // lock released; no force needed
    again.close();
  }

  // ── torn final line tolerated; terminated garbage throws ───────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const a = await createChunk(ws.ctxFor('human:asa'), { text: 'durable' });
    ws.close();
    appendFileSync(sub(root, 'log.jsonl'), '{"id":"cmt_torn","parentIds'); // crash mid-append

    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.chunks.size, 1);
    assert.equal(renderChunk(ws2.state, a.chunkId), 'durable');
    await createChunk(ws2.ctxFor('human:asa'), { text: 'after the tear' }); // append stays line-aligned
    ws2.close();
    const ws3 = await openWorkspace(root);
    assert.equal(ws3.state.chunks.size, 2);
    ws3.close();

    appendFileSync(sub(root, 'log.jsonl'), 'not json\n'); // terminated line = corruption
    await assert.rejects(() => openWorkspace(root), /corrupt/);
    assert.ok(!existsSync(sub(root, 'lock')), 'failed open releases the lock');
  }

  // ── blob files are content-addressed ───────────────────────────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const text = 'content addressed payload';
    await createChunk(ws.ctxFor('human:asa'), { text });
    const hash = await blobHashOf(MEDIA_TEXT, text);
    const path = sub(root, 'blobs', hash.slice(0, 2), hash);
    assert.ok(existsSync(path), `blob file missing at ${path}`);
    const stored = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(stored.mediaType, MEDIA_TEXT);
    assert.equal(stored.text, text);
    ws.close();
  }

  // ── auto-snapshot on the 50th append ───────────────────────────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const ctx = ws.ctxFor('human:asa');
    for (let i = 0; i < 49; i++) await createChunk(ctx, { text: `c${i}` });
    assert.ok(!existsSync(sub(root, 'snapshot.json')), 'no snapshot before 50 appends');
    await createChunk(ctx, { text: 'c49' });
    const snap = JSON.parse(readFileSync(sub(root, 'snapshot.json'), 'utf8'));
    assert.equal(snap.coveredCommits, 50);
    crash(root); // snapshot alone must reconstruct
    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.chunks.size, 50);
    ws2.close();
  }
} finally {
  for (const d of roots) rmSync(d, { recursive: true, force: true });
}

console.log('store-fs OK —', roots.length, 'workspaces exercised');
