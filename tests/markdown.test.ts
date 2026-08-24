import assert from 'node:assert';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { childOccurrences, currentRevision, emptyState, renderChunk, revisionText } from '../src/kernel/state';
import { acceptProposal, promoteExtract, revise, transclude, type TxCtx } from '../src/kernel/tx';
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

// Cost guards (host/similarity.ts). Distance is at least the length
// difference, so a pair that far apart provably cannot reach the driver's 0.5
// threshold: zero here is a floor, not a measurement.
assert.equal(similarity('a'.repeat(10), 'a'.repeat(100)), 0);
// Above the exact cap the score comes from a bounded sample, so one comparison
// stays bounded however large the blocks are. Unguarded, this pair is a
// 200k x 200k DP — minutes of blocked event loop inside a reconcile that the
// dev server runs with every other request queued behind it.
{
  const huge = 'lorem ipsum dolor sit amet '.repeat(8000);
  const editedTail = `${huge.slice(0, huge.length - 40)}tail rewritten outside of substrate xx`;
  const started = performance.now();
  const score = similarity(huge, editedTail);
  const elapsed = performance.now() - started;
  assert.ok(elapsed < 1000, `similarity must stay bounded; took ${Math.round(elapsed)}ms`);
  assert.ok(score >= 0.5, 'a large block with a small edit still matches its own history');
}

const root = mkdtempSync(join(tmpdir(), 'substrate-md-'));
try {
  const state = emptyState();
  const dctx: TxCtx = { state, actorId: 'driver:fs' };
  const hctx: TxCtx = { state, actorId: 'human:asa' };
  const rel = 'notes/demo.md';
  const readSc = (): MarkdownSidecar => JSON.parse(readFileSync(sidecarPath(root, rel), 'utf8'));
  const putSource = (relPath: string, text: string): string => {
    const abs = join(root, relPath);
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, text);
    return abs;
  };
  const assertNoAtomicTemps = (path: string): void => {
    assert.deepEqual(
      readdirSync(dirname(path)).filter((name) => name.startsWith('.substrate-write-')),
      [],
      `atomic replacement left a temporary file beside ${path}`,
    );
  };

  const A = '# Title';
  const B = 'para one\nline two';
  const C = '```js\ncode\n```';
  const D = '- item a\n- item b';
  const mk = (bs: string[]) => `${bs.join('\n\n')}\n`;

  // canonical round-trip: import then project is byte-identical
  const md1 = mk([A, B, C, D]);
  const imported = await importMarkdownFile(dctx, {
    workspaceRoot: root,
    relPath: rel,
    text: md1,
    operationParams: { sourceId: 'source_demo', observationId: 'observation_demo', blocks: 999 },
  });
  const { docChunkId, blockChunkIds } = imported;
  assert.equal(imported.commitId, imported.commit.id, 'driver returns the exact persisted import commit');
  assert.equal(imported.operationId, imported.commit.operation.id, 'driver returns the exact persisted import operation');
  assert.equal(imported.commit.id, state.head);
  assert.deepEqual(state.operations.get(imported.operationId)?.params, {
    sourceId: 'source_demo',
    observationId: 'observation_demo',
    blocks: 4,
  });
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

  // Projection atomically replaces a source that still matches the manifest
  // and reconverges the sidecar. The source fixture starts at the last file
  // snapshot imported before the two-sided reconciliation proposal.
  const relAbs = putSource(rel, md3);
  await writeProjection(dctx, { workspaceRoot: root, relPath: rel });
  const onDisk = readFileSync(relAbs, 'utf8');
  assert.equal(onDisk, projectMarkdown(state, docChunkId));
  assert.equal((await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: rel, text: onDisk })).action, 'noop');
  assertNoAtomicTemps(relAbs);
  assertNoAtomicTemps(sidecarPath(root, rel));

  // A source changed since the sidecar snapshot is never overwritten, and a
  // failed precondition does not advance the sidecar.
  const externalChange = `${onDisk.trimEnd()}\n\nexternal edit not yet synchronized\n`;
  putSource(rel, externalChange);
  const projectionSidecarBeforeRefusal = readFileSync(sidecarPath(root, rel), 'utf8');
  await assert.rejects(
    writeProjection(dctx, { workspaceRoot: root, relPath: rel }),
    /source changed since its last import or projection/,
  );
  assert.equal(readFileSync(relAbs, 'utf8'), externalChange, 'projection refusal preserves the external edit');
  assert.equal(
    readFileSync(sidecarPath(root, rel), 'utf8'),
    projectionSidecarBeforeRefusal,
    'projection refusal leaves the sidecar untouched',
  );

  // A sidecar alone is not authority to recreate a vanished source.
  const missingRel = 'notes/missing.md';
  await importMarkdownFile(dctx, { workspaceRoot: root, relPath: missingRel, text: '# Missing\n' });
  const missingSidecarBefore = readFileSync(sidecarPath(root, missingRel), 'utf8');
  await assert.rejects(
    writeProjection(dctx, { workspaceRoot: root, relPath: missingRel }),
    /source file is missing/,
  );
  assert.equal(existsSync(join(root, missingRel)), false, 'projection does not recreate a missing source');
  assert.equal(readFileSync(sidecarPath(root, missingRel), 'utf8'), missingSidecarBefore);

  // Recovery from a source-replaced/sidecar-stale partial attempt is safe: if
  // disk already equals the intended projection, retry advances the manifest.
  const recoveryRel = 'notes/recovery.md';
  const recoveryText = '# Recovery\n';
  const recovery = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: recoveryRel, text: recoveryText });
  putSource(recoveryRel, recoveryText);
  await revise(hctx, { chunkId: recovery.blockChunkIds[0], text: '# Recovered' });
  const intendedRecovery = projectMarkdown(state, recovery.docChunkId);
  putSource(recoveryRel, intendedRecovery); // prior attempt replaced only the source
  const staleRecoverySidecar = readFileSync(sidecarPath(root, recoveryRel), 'utf8');
  await writeProjection(dctx, { workspaceRoot: root, relPath: recoveryRel });
  const recoveredSidecar = readFileSync(sidecarPath(root, recoveryRel), 'utf8');
  assert.notEqual(recoveredSidecar, staleRecoverySidecar, 'retry advances the stale sidecar');
  assert.equal(readFileSync(join(root, recoveryRel), 'utf8'), intendedRecovery);
  assertNoAtomicTemps(join(root, recoveryRel));
  assertNoAtomicTemps(sidecarPath(root, recoveryRel));

  // Extracted structure projects through the occurrence-aware renderer. An
  // external edit cannot revise the composite's JSON blob directly: it becomes
  // an explicit flatten proposal that severs the now-replaced child structure.
  const extractRel = 'notes/extracted.md';
  const extractText = mk(['# Extracted', 'Second paragraph with a phrase to extract inside it.']);
  const extractedDoc = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: extractRel, text: extractText });
  const compositeChunkId = extractedDoc.blockChunkIds[1];
  const flatRev = currentRevision(state, compositeChunkId);
  const flatText = revisionText(state, flatRev.id);
  const phrase = 'a phrase to extract';
  const phraseStart = flatText.indexOf(phrase);
  await promoteExtract(hctx, {
    span: { revisionId: flatRev.id, method: 'raw@1', start: phraseStart, end: phraseStart + phrase.length },
  });
  assert.equal(projectMarkdown(state, extractedDoc.docChunkId), extractText, 'composite projection preserves rendered Markdown');
  putSource(extractRel, extractText);
  await writeProjection(dctx, { workspaceRoot: root, relPath: extractRel });
  assert.equal(readFileSync(join(root, extractRel), 'utf8'), extractText, 'projection never writes composite join JSON');
  const extractSc = JSON.parse(readFileSync(sidecarPath(root, extractRel), 'utf8')) as MarkdownSidecar;
  const compositeEntry = extractSc.blocks.find((b) => b.chunkId === compositeChunkId)!;
  assert.equal(compositeEntry.policy, 'flatten-composite');
  assert.deepEqual(compositeEntry.occurrencePath, [compositeEntry.occurrenceId]);
  assert.equal(compositeEntry.projectedText, flatText);

  const flattenedText = extractText.replace('phrase to extract', 'phrase edited outside');
  const compositeHeadBefore = currentRevision(state, compositeChunkId).id;
  const flattenResult = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: extractRel, text: flattenedText });
  assert.equal(flattenResult.action, 'proposal');
  assert.equal(currentRevision(state, compositeChunkId).id, compositeHeadBefore, 'composite is untouched before proposal acceptance');
  const flattenProposal = state.proposals.get(flattenResult.proposalId!)!;
  assert.equal(flattenProposal.payload[0].op, 'revise');
  assert.equal(flattenProposal.payload[0].op === 'revise' && flattenProposal.payload[0].mediaType, 'text/markdown');
  assert.ok(flattenProposal.payload.slice(1).every((change) => change.op === 'sever'));
  assert.ok((await acceptProposal(hctx, { proposalId: flattenResult.proposalId! })).applied);
  assert.equal(projectMarkdown(state, extractedDoc.docChunkId), flattenedText);
  assert.equal(
    (await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: extractRel, text: flattenedText })).action,
    'fast-forward',
    'accepted structural proposal converges the manifest',
  );

  // Projected transclusions are readable but never writable authority over the
  // source. An external edit proposes a derived local copy in the target and
  // severs only the target occurrence.
  const sourceRel = 'notes/source.md';
  const targetRel = 'notes/target.md';
  const sourceText = mk(['# Source', 'A quotable sentence that lives in the source.']);
  const targetText = mk(['# Target', 'Local target paragraph.']);
  const sourceDoc = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: sourceRel, text: sourceText });
  const targetDoc = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: targetRel, text: targetText });
  const sourceChunkId = sourceDoc.blockChunkIds[1];
  const sourceRevisionId = currentRevision(state, sourceChunkId).id;
  const transclusion = transclude(hctx, { containerId: targetDoc.docChunkId, sourceChunkId });
  putSource(targetRel, targetText);
  const targetProjection = await writeProjection(dctx, { workspaceRoot: root, relPath: targetRel });
  const targetSc = JSON.parse(readFileSync(sidecarPath(root, targetRel), 'utf8')) as MarkdownSidecar;
  const transcludedEntry = targetSc.blocks.find((b) => b.occurrenceId === transclusion.occurrenceId)!;
  assert.equal(transcludedEntry.mode, 'transclude');
  assert.equal(transcludedEntry.pin, sourceRevisionId);
  assert.equal(transcludedEntry.sourceRevisionId, sourceRevisionId);
  assert.equal(transcludedEntry.policy, 'detach-transclusion');

  const detachedText = targetProjection.text.replace(
    'A quotable sentence that lives in the source.',
    'A quotable sentence rewritten only in the target.',
  );
  const detachResult = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: targetRel, text: detachedText });
  assert.equal(detachResult.action, 'proposal');
  assert.equal(currentRevision(state, sourceChunkId).id, sourceRevisionId, 'reconcile never revises a transclusion source');
  const detachProposal = state.proposals.get(detachResult.proposalId!)!;
  assert.deepEqual(detachProposal.payload.map((change) => change.op), ['create', 'place', 'sever']);
  const detachedCreate = detachProposal.payload[0];
  assert.equal(detachedCreate.op === 'create' && detachedCreate.derivedFrom?.sourceRevisionId, sourceRevisionId);
  assert.ok((await acceptProposal(hctx, { proposalId: detachResult.proposalId! })).applied);
  assert.equal(currentRevision(state, sourceChunkId).id, sourceRevisionId, 'accepting detach leaves source history untouched');
  assert.equal(renderChunk(state, targetDoc.docChunkId), detachedText.trimEnd());
  assert.equal(
    (await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: targetRel, text: detachedText })).action,
    'fast-forward',
    'accepted detach proposal converges the manifest',
  );

  // One external pass carrying all three leaf changes at once: an edit, an
  // insertion between existing blocks, and a deletion. Identity survives the
  // edit, the insertion lands in file order, and the deletion is never enacted
  // by the driver — it is proposed and waits for a human.
  const mixRel = 'notes/mixed.md';
  const P1 = '# Mixed';
  const P2 = 'alpha paragraph';
  const P3 = 'beta paragraph';
  const P4 = 'gamma paragraph';
  const mixed = await importMarkdownFile(dctx, { workspaceRoot: root, relPath: mixRel, text: mk([P1, P2, P3, P4]) });
  const [c1, c2, c3, c4] = mixed.blockChunkIds;
  const revsBefore = mixed.blockChunkIds.map((id) => currentRevision(state, id).id);

  const P2edited = 'alpha paragraph edited outside';
  const Padded = 'delta paragraph added outside';
  const mixedText = mk([P1, P2edited, Padded, P3]);
  const mixResult = await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: mixRel, text: mixedText });
  assert.equal(mixResult.action, 'fast-forward');

  assert.equal(currentRevision(state, c1).id, revsBefore[0], 'untouched block keeps its revision');
  assert.equal(currentRevision(state, c3).id, revsBefore[2], 'untouched block keeps its revision');
  assert.notEqual(currentRevision(state, c2).id, revsBefore[1], 'edited block advances');
  assert.ok(state.chunks.has(c2), 'edited block keeps its identity');
  assert.equal(revisionText(state, currentRevision(state, c2).id), P2edited);
  assert.equal(state.revisions.get(currentRevision(state, c2).id)!.createdBy, 'driver:fs');

  const mixOccs = childOccurrences(state, mixed.docChunkId);
  const mixTexts = mixOccs.map((occ) => revisionText(state, currentRevision(state, occ.chunkId).id));
  assert.deepEqual(mixTexts, [P1, P2edited, Padded, P3, P4], 'insertion lands in file order; deletion still stands');
  assert.ok(!mixed.blockChunkIds.includes(mixOccs[2].chunkId), 'the inserted block is a new chunk');

  assert.ok(mixResult.proposalId, 'the deletion is proposed, not enacted');
  const mixProposal = state.proposals.get(mixResult.proposalId!)!;
  assert.deepEqual(mixProposal.payload.map((change) => change.op), ['sever']);
  assert.equal(
    mixProposal.payload[0].op === 'sever' && state.occurrences.get(mixProposal.payload[0].occurrenceId)!.chunkId,
    c4,
  );
  assert.ok((await acceptProposal(hctx, { proposalId: mixResult.proposalId! })).applied);
  assert.equal(projectMarkdown(state, mixed.docChunkId), mixedText, 'the document now reads exactly as the file');
  assert.ok(state.chunks.has(c4) && !state.chunks.get(c4)!.tombstoned, 'severing an occurrence never deletes the chunk');
  assert.equal(
    (await reconcileMarkdownFile(dctx, { workspaceRoot: root, relPath: mixRel, text: mixedText })).action,
    'noop',
    'the round trip settles',
  );

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
