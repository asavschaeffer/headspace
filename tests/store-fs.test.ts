// Filesystem workspace store: durability, replay, locking, crash artifacts.
// Each block owns a fresh tmp workspace; a deleted lock file simulates a crash
// (close() always snapshots, so crashing is the only way to exercise replay).
import assert from 'node:assert';
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openWorkspace } from '../src/host/store-fs';
import { blobHashOf } from '../src/kernel/hash';
import { renderChunk } from '../src/kernel/state';
import { createChunk, propose, rejectProposal, revise } from '../src/kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_TEXT } from '../src/kernel/types';

const roots: string[] = [];
const freshRoot = () => {
  const d = mkdtempSync(join(tmpdir(), 'headspace-store-'));
  roots.push(d);
  return d;
};
const dataPath = (root: string, ...parts: string[]) => join(root, '.headspace', ...parts);
const crash = (root: string) => rmSync(dataPath(root, 'lock')); // died without close()

try {
  // ── write then reopen: log-only materialize, then snapshot path ────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    assert.equal(ws.dataDir, dataPath(root));
    assert.ok(existsSync(dataPath(root)), 'new durable state is created under .headspace');
    const ctx = ws.ctxFor('human:asa');
    const doc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
    await createChunk(ctx, { text: 'the lake is still', containerId: doc.chunkId });
    const b2 = await createChunk(ctx, { text: 'the sky above it', containerId: doc.chunkId });
    await revise(ctx, { chunkId: b2.chunkId, text: 'the sky above it burns' });
    const rendered = renderChunk(ctx.state, doc.chunkId);
    const chunkCount = ctx.state.chunks.size;
    const occCount = ctx.state.occurrences.size;

    crash(root);
    assert.ok(!existsSync(dataPath(root, 'snapshot.json')), 'no snapshot was written before the crash');
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

    const snap = JSON.parse(readFileSync(dataPath(root, 'snapshot.json'), 'utf8'));
    assert.equal(snap.schemaVersion, 1, 'snapshots declare the current schema explicitly');
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

  // ── lock: live conflict throws, stale lock recovered, force takes over ─────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    await assert.rejects(() => openWorkspace(root), /locked by running pid/);
    const forced = await openWorkspace(root, { force: true });
    forced.close();
    const again = await openWorkspace(root); // lock released; no force needed
    again.close();
    // A lock left by a dead process is a crash artifact, not a writer.
    writeFileSync(dataPath(root, 'lock'), '999999999');
    const recovered = await openWorkspace(root);
    recovered.close();
    void ws;
  }

  // ── latest snapshot schema and proposal provenance are mandatory ─────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const chunk = await createChunk(ws.ctxFor('human:asa'), { text: 'proposal target' });
    const proposed = propose(ws.ctxFor('agent:test'), {
      kind: 'suggested-edit',
      basisRevisionIds: [chunk.revisionId],
      targetChunkIds: [chunk.chunkId],
      payload: [{ op: 'revise', chunkId: chunk.chunkId, text: 'proposed text' }],
    });
    rejectProposal(ws.ctxFor('human:reviewer'), { proposalId: proposed.proposalId });
    ws.close();

    const snapshotPath = dataPath(root, 'snapshot.json');
    const current = JSON.parse(readFileSync(snapshotPath, 'utf8')) as {
      schemaVersion?: number;
      state: { proposals: Array<{ operationId: string; resolution?: { operationId: string } }> };
    };
    const pristine = structuredClone(current);
    delete current.schemaVersion;
    writeFileSync(snapshotPath, JSON.stringify(current));
    await assert.rejects(() => openWorkspace(root), /unsupported workspace snapshot schema: undefined/);
    assert.ok(!existsSync(dataPath(root, 'lock')), 'rejected old snapshot releases the workspace lock');

    current.schemaVersion = 1;
    current.state.proposals[0].operationId = 'op_missing';
    writeFileSync(snapshotPath, JSON.stringify(current));
    await assert.rejects(() => openWorkspace(root), /does not resolve to its matching propose operation/);
    assert.ok(!existsSync(dataPath(root, 'lock')), 'rejected malformed snapshot releases the workspace lock');

    const malformedResolution = structuredClone(pristine);
    assert.ok(malformedResolution.state.proposals[0].resolution);
    malformedResolution.state.proposals[0].resolution.operationId = 'op_missing';
    writeFileSync(snapshotPath, JSON.stringify(malformedResolution));
    await assert.rejects(() => openWorkspace(root), /resolution does not resolve consistently/);
    assert.ok(!existsSync(dataPath(root, 'lock')), 'rejected resolution linkage releases the workspace lock');
  }

  // ── torn final line tolerated; terminated garbage throws ───────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const a = await createChunk(ws.ctxFor('human:asa'), { text: 'durable' });
    ws.close();
    appendFileSync(dataPath(root, 'log.jsonl'), '{"id":"cmt_torn","parentIds'); // crash mid-append

    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.chunks.size, 1);
    assert.equal(renderChunk(ws2.state, a.chunkId), 'durable');
    await createChunk(ws2.ctxFor('human:asa'), { text: 'after the tear' }); // append stays line-aligned
    ws2.close();
    const ws3 = await openWorkspace(root);
    assert.equal(ws3.state.chunks.size, 2);
    ws3.close();

    appendFileSync(dataPath(root, 'log.jsonl'), 'not json\n'); // terminated line = corruption
    await assert.rejects(() => openWorkspace(root), /corrupt/);
    assert.ok(!existsSync(dataPath(root, 'lock')), 'failed open releases the lock');
  }

  // ── blob files are content-addressed ───────────────────────────────────────
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    const text = 'content addressed payload';
    await createChunk(ws.ctxFor('human:asa'), { text });
    const hash = await blobHashOf(MEDIA_TEXT, text);
    const path = dataPath(root, 'blobs', hash.slice(0, 2), hash);
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
    assert.ok(!existsSync(dataPath(root, 'snapshot.json')), 'no snapshot before 50 appends');
    await createChunk(ctx, { text: 'c49' });
    const snap = JSON.parse(readFileSync(dataPath(root, 'snapshot.json'), 'utf8'));
    assert.equal(snap.coveredCommits, 50);
    crash(root); // snapshot alone must reconstruct
    const ws2 = await openWorkspace(root);
    assert.equal(ws2.state.chunks.size, 50);
    ws2.close();
  }

  // ── a failed append is a non-event, and the store stops writing ────────────
  // A directory where log.jsonl belongs makes every append throw. The point is
  // what memory does about it: nothing. State must stay exactly where the log
  // is, because a snapshot records a LINE OFFSET into that log — state one
  // commit ahead would tell the next open to skip a line that is not there.
  {
    const root = freshRoot();
    const ws = await openWorkspace(root);
    mkdirSync(dataPath(root, 'log.jsonl')); // nothing can be appended to a directory
    const ctx = ws.ctxFor('human:asa');

    await assert.rejects(() => createChunk(ctx, { text: 'never durable' }));
    assert.equal(ws.state.commitCount, 0, 'a failed append must not count as a commit');
    assert.equal(ws.state.chunks.size, 0, 'a failed append must not mint a chunk');
    assert.equal(ws.state.head, null, 'a failed append must not advance head');

    // Past the first failure the log may hold a partial line. Appending beyond
    // it would bury the tear mid-log, where recovery cannot tell it from
    // corruption — so the store refuses, and refuses to snapshot over it.
    await assert.rejects(() => createChunk(ctx, { text: 'after the failure' }), /not writable/);
    assert.throws(() => ws.saveSnapshot(), /refusing to snapshot an unwritable workspace/);

    // It still releases the lock: a store that cannot write must not also hold
    // the workspace hostage.
    ws.close();
    assert.ok(!existsSync(dataPath(root, 'lock')), 'close releases the lock even after a write failure');
    assert.ok(!existsSync(dataPath(root, 'snapshot.json')), 'no snapshot claims commits the log never took');
  }

} finally {
  for (const d of roots) rmSync(d, { recursive: true, force: true });
}

console.log('store-fs OK —', roots.length, 'workspaces exercised');
