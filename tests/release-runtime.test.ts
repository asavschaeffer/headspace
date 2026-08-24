import assert from 'node:assert';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { OFFLINE_COLLABORATOR, stubCompleter } from '../src/collaboration/stub';
import { deserializeState } from '../src/kernel/serialize';
import { generateProposal, select } from '../src/kernel/select';
import { childOccurrences, currentRevision, renderChunk, revisionText } from '../src/kernel/state';
import { acceptProposal, revise, type TxCtx } from '../src/kernel/tx';
import type { Commit } from '../src/kernel/types';
import { createReleaseServer, startReleaseServer, type ReleaseServer } from '../src/host/serve';

interface StatePayload {
  state: Parameters<typeof deserializeState>[0];
  bindings: { docChunkId: string; relPath: string }[];
}

function rawGet(url: string, headers: Record<string, string>): Promise<{ body: string; status: number }> {
  return new Promise((resolveRequest, rejectRequest) => {
    const req = request(url, { headers }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk: string) => {
        body += chunk;
      });
      res.on('end', () => resolveRequest({ body, status: res.statusCode ?? 0 }));
    });
    req.on('error', rejectRequest);
    req.end();
  });
}

const sandbox = mkdtempSync(join(tmpdir(), 'headspace-release-runtime-'));
const workspace = join(sandbox, 'mixed-workspace');
const dist = join(sandbox, 'dist');
const secret = 'outside-static-root';
const emptyDist = join(sandbox, 'empty-dist');
const distFile = join(sandbox, 'not-a-dist-directory');
const outsideAssets = join(sandbox, 'outside-assets');
const escapedIndexDist = join(sandbox, 'escaped-index-dist');
const corruptWorkspace = join(sandbox, 'corrupt-workspace');
mkdirSync(join(workspace, 'notes'), { recursive: true });
mkdirSync(join(workspace, 'references'), { recursive: true });
mkdirSync(join(dist, 'assets'), { recursive: true });
mkdirSync(emptyDist, { recursive: true });
mkdirSync(outsideAssets, { recursive: true });
mkdirSync(escapedIndexDist, { recursive: true });
mkdirSync(join(corruptWorkspace, '.headspace'), { recursive: true });
writeFileSync(join(workspace, 'notes', 'focus.md'), '# Release focus\n\nA durable local thought.\n');
writeFileSync(join(workspace, 'references', 'plain.txt'), 'A second ingestible representation.\n');
writeFileSync(join(workspace, 'references', 'opaque.bin'), Buffer.from([0, 1, 2, 3]));
writeFileSync(join(dist, 'index.html'), '<!doctype html><main>Headspace release shell</main>');
writeFileSync(join(dist, 'assets', 'app.js'), 'globalThis.headspaceRelease = true;');
writeFileSync(join(sandbox, 'secret.txt'), secret);
writeFileSync(join(outsideAssets, 'secret.txt'), secret);
writeFileSync(distFile, 'not a directory');
writeFileSync(join(corruptWorkspace, '.headspace', 'log.jsonl'), 'not a commit\n');

const directoryLinkType = process.platform === 'win32' ? 'junction' : 'dir';
let directoryLinksAvailable = false;
try {
  symlinkSync(outsideAssets, join(escapedIndexDist, 'index.html'), directoryLinkType);
  symlinkSync(outsideAssets, join(dist, 'assets', 'outside'), directoryLinkType);
  directoryLinksAvailable = true;
} catch (error) {
  const code = (error as NodeJS.ErrnoException).code;
  if (code !== 'EPERM' && code !== 'EACCES' && code !== 'ENOSYS' && code !== 'ENOTSUP') throw error;
}

let runtime: ReleaseServer | null = null;
try {
  assert.throws(
    () => createReleaseServer({ root: join(sandbox, 'missing-workspace'), distDir: dist }),
    /workspace root does not exist/,
  );
  assert.throws(
    () => createReleaseServer({ root: workspace, distDir: distFile }),
    /built application directory is not a directory/,
  );
  assert.throws(
    () => createReleaseServer({ root: workspace, distDir: emptyDist }),
    /built application entry does not exist/,
  );
  if (directoryLinksAvailable) {
    assert.throws(
      () => createReleaseServer({ root: workspace, distDir: escapedIndexDist }),
      /entry is not a confined regular file/,
    );
  }

  const corruptRuntime = createReleaseServer({
    root: corruptWorkspace,
    distDir: dist,
    host: '127.0.0.1',
    port: 0,
    collaborators: [],
  });
  await assert.rejects(
    corruptRuntime.listen(),
    /log\.jsonl corrupt/,
    'corrupt durable state must fail before the release host binds',
  );
  assert.equal(corruptRuntime.server.listening, false);
  await corruptRuntime.close();

  assert.throws(
    () => createReleaseServer({
      root: workspace,
      distDir: dist,
      host: '0.0.0.0',
      port: 0,
      collaborators: [],
    }),
    /loopback-only/,
    'the unauthenticated 0.0.1 host must never bind to the LAN',
  );

  runtime = await startReleaseServer({
    root: workspace,
    distDir: dist,
    contentDirs: ['.'],
    host: '127.0.0.1',
    port: 0,
    collaborators: [],
  });
  const firstBase = runtime.address!.url;

  const lockedRuntime = createReleaseServer({
    root: workspace,
    distDir: dist,
    contentDirs: ['.'],
    host: '127.0.0.1',
    port: 0,
    collaborators: [],
  });
  await assert.rejects(
    lockedRuntime.listen(),
    /locked by running pid/,
    'listen must validate workspace ownership before binding a socket',
  );
  assert.equal(lockedRuntime.server.listening, false);
  await lockedRuntime.close();

  const catalogPath = join(workspace, '.headspace', 'ingestion.json');
  const catalogBeforeRejectedRequests = readFileSync(catalogPath, 'utf8');

  const rebound = await rawGet(`${firstBase}/api/state`, { host: 'evil.example' });
  assert.equal(rebound.status, 421);
  assert.doesNotMatch(rebound.body, /durable local thought/);

  const crossOriginMutation = await fetch(`${firstBase}/api/ingest`, {
    method: 'POST',
    headers: { origin: 'https://evil.example' },
  });
  assert.equal(crossOriginMutation.status, 403);
  assert.equal(
    readFileSync(catalogPath, 'utf8'),
    catalogBeforeRejectedRequests,
    'rejected browser requests cannot mutate workspace state',
  );

  const sameOriginState = await fetch(`${firstBase}/api/state`, {
    headers: { origin: firstBase },
  });
  assert.equal(sameOriginState.status, 200);

  const shell = await fetch(`${firstBase}/`);
  assert.equal(shell.status, 200);
  assert.match(shell.headers.get('content-type') ?? '', /^text\/html/);
  assert.equal(shell.headers.get('x-frame-options'), 'DENY');
  assert.equal(shell.headers.get('content-security-policy'), "frame-ancestors 'none'");
  assert.equal(shell.headers.get('referrer-policy'), 'no-referrer');
  assert.match(await shell.text(), /Headspace release shell/);

  const asset = await fetch(`${firstBase}/assets/app.js`);
  assert.equal(asset.status, 200);
  assert.match(asset.headers.get('content-type') ?? '', /^text\/javascript/);
  assert.match(await asset.text(), /headspaceRelease/);

  const missingAsset = await fetch(`${firstBase}/assets/missing.js`);
  assert.equal(missingAsset.status, 404);
  assert.doesNotMatch(await missingAsset.text(), /Headspace release shell/);

  const clientRoute = await fetch(`${firstBase}/star/a-client-route`);
  assert.equal(clientRoute.status, 200);
  assert.match(await clientRoute.text(), /Headspace release shell/);

  const escaped = await fetch(`${firstBase}/..%2fsecret.txt`);
  assert.equal(escaped.status, 403);
  assert.doesNotMatch(await escaped.text(), new RegExp(secret));

  const escapedBackslash = await fetch(`${firstBase}/..%5csecret.txt`);
  assert.equal(escapedBackslash.status, 403);
  assert.doesNotMatch(await escapedBackslash.text(), new RegExp(secret));

  if (directoryLinksAvailable) {
    const linkedEscape = await fetch(`${firstBase}/assets/outside/secret.txt`);
    assert.equal(linkedEscape.status, 404);
    assert.doesNotMatch(await linkedEscape.text(), new RegExp(secret));
  }

  // Exercise the explicit release ingest endpoint before reconstructing the
  // browser's local materialized state.
  const removedSyncAlias = await fetch(`${firstBase}/api/sync`, { method: 'POST' });
  assert.equal(removedSyncAlias.status, 404, 'the pre-release sync route alias is not exposed');
  const ingestion = await fetch(`${firstBase}/api/ingest`, { method: 'POST' });
  if (ingestion.status !== 200) {
    assert.fail(`release ingest failed (${ingestion.status}): ${await ingestion.text()}`);
  }
  const ingestionPayload = await ingestion.json() as {
    report: { ingestion: { items: { observation: { relPath: string }; status: string }[] } };
  };
  assert.ok(
    ingestionPayload.report.ingestion.items.some(
      (item) => item.observation.relPath === 'notes/focus.md' && ['imported', 'unchanged'].includes(item.status),
    ),
  );
  assert.ok(
    ingestionPayload.report.ingestion.items.some(
      (item) => item.observation.relPath === 'references/plain.txt',
    ),
    'the mixed workspace is scanned through the ingestion seam',
  );

  const stateResponse = await fetch(`${firstBase}/api/state`);
  assert.equal(stateResponse.status, 200);
  const payload = await stateResponse.json() as StatePayload;
  const state = deserializeState(payload.state);
  const binding = payload.bindings.find((item) => item.relPath === 'notes/focus.md');
  assert.ok(binding, 'Markdown source is bound to a focusable document');
  const plainBinding = payload.bindings.find((item) => item.relPath === 'references/plain.txt');
  assert.ok(plainBinding, 'plain-text source is bound through its native adapter');
  assert.match(renderChunk(state, plainBinding.docChunkId), /second ingestible representation/);

  const selected = select(state, binding.docChunkId);
  assert.equal(selected[0]?.role, 'focus');
  assert.equal(selected[0]?.chunkId, binding.docChunkId);
  assert.match(selected[0]?.text ?? '', /durable local thought/);

  // Reproduce the browser contract: mutate a local materialization, retain the
  // exact generated commits, then submit those commits to the authoritative
  // release host for validation and durable append.
  const commits: Commit[] = [];
  const authorCtx: TxCtx = {
    state,
    actorId: 'human:release-author',
    onCommit: (commit) => commits.push(commit),
  };
  const editableOccurrence = childOccurrences(state, binding.docChunkId).find(
    (occurrence) => revisionText(state, currentRevision(state, occurrence.chunkId).id).includes('durable local thought'),
  );
  assert.ok(editableOccurrence, 'the ingested Markdown paragraph is an addressable editable part');
  const editedChunkId = editableOccurrence.chunkId;
  const priorEditRevision = currentRevision(state, editedChunkId);
  const headBeforeEdit = state.head;
  const editedText = 'A durable local thought, edited through the browser commit boundary.';
  const edit = await revise(authorCtx, { chunkId: editedChunkId, text: editedText });
  assert.equal(edit.commit.operation.kind, 'revise');
  assert.deepEqual(edit.commit.parentIds, headBeforeEdit ? [headBeforeEdit] : []);

  const generated = await generateProposal(authorCtx, {
    focusChunkId: binding.docChunkId,
    instruction: 'prove the packaged collaboration loop',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });
  const proposalBeforeAcceptance = state.proposals.get(generated.proposalId)!;
  const proposalOperation = state.operations.get(proposalBeforeAcceptance.operationId!)!;
  assert.equal(proposalBeforeAcceptance.createdBy, OFFLINE_COLLABORATOR.actorId);
  assert.ok(proposalOperation.inputRevisionIds.length > 0);

  const accepted = await acceptProposal(
    { ...authorCtx, actorId: 'human:release-reviewer' },
    { proposalId: generated.proposalId },
  );
  assert.ok(accepted.applied);
  const createdChunkId = accepted.createdChunkIds[0];
  assert.ok(createdChunkId);
  assert.match(renderChunk(state, binding.docChunkId), /prove the packaged collaboration loop/);

  const commitResponse = await fetch(`${firstBase}/api/commits`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ commits }),
  });
  assert.equal(commitResponse.status, 200, await commitResponse.text());

  // Exercise the same explicit projection endpoint as Star. A clean source is
  // replaced, while a later external edit is preserved and reported as a
  // conflict rather than overwritten.
  const projection = await fetch(`${firstBase}/api/project`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ relPath: binding.relPath }),
  });
  assert.equal(projection.status, 200, await projection.text());
  const projectedText = readFileSync(join(workspace, binding.relPath), 'utf8');
  assert.ok(projectedText.includes(editedText));
  assert.match(projectedText, /prove the packaged collaboration loop/);

  const externalEdit = '# Release focus\n\nExternal edit that Headspace must not overwrite.\n';
  writeFileSync(join(workspace, binding.relPath), externalEdit);
  const projectionConflict = await fetch(`${firstBase}/api/project`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ relPath: binding.relPath }),
  });
  assert.equal(projectionConflict.status, 409);
  const conflictBody = await projectionConflict.json() as { code?: string; error?: string };
  assert.equal(conflictBody.code, 'projection-conflict');
  assert.match(conflictBody.error ?? '', /source changed since its last import or projection/);
  assert.equal(readFileSync(join(workspace, binding.relPath), 'utf8'), externalEdit);

  await runtime.close();
  await assert.rejects(runtime.listen(), /closed and cannot listen again/);
  assert.equal(runtime.server.listening, false);
  runtime = null;

  // A fresh HTTP host must reopen the filesystem store, not reuse the client
  // state or the first server's in-memory workspace.
  runtime = await startReleaseServer({
    root: workspace,
    distDir: dist,
    contentDirs: ['.'],
    host: '127.0.0.1',
    port: 0,
    collaborators: [],
  });
  const restartedResponse = await fetch(`${runtime.address!.url}/api/state`);
  assert.equal(restartedResponse.status, 200);
  const restartedPayload = await restartedResponse.json() as StatePayload;
  const restarted = deserializeState(restartedPayload.state);
  const durableProposal = restarted.proposals.get(generated.proposalId)!;
  assert.equal(durableProposal.status, 'accepted');
  assert.equal(durableProposal.createdBy, OFFLINE_COLLABORATOR.actorId, 'collaborator authorship persists');
  assert.equal(durableProposal.resolution?.by, 'human:release-reviewer', 'human acceptance persists');
  assert.deepEqual(
    restarted.operations.get(durableProposal.operationId!)?.inputRevisionIds,
    proposalOperation.inputRevisionIds,
    'the exact dispatched revision inputs persist',
  );
  assert.equal(currentRevision(restarted, createdChunkId).createdBy, OFFLINE_COLLABORATOR.actorId);
  assert.match(renderChunk(restarted, binding.docChunkId), /prove the packaged collaboration loop/);
  assert.match(renderChunk(restarted, plainBinding.docChunkId), /second ingestible representation/);

  const durableEdit = currentRevision(restarted, editedChunkId);
  assert.equal(durableEdit.id, edit.revisionId, 'the edited revision remains current after a fresh host restart');
  assert.equal(revisionText(restarted, durableEdit.id), editedText);
  assert.ok(restarted.revisions.has(priorEditRevision.id), 'the pre-edit revision remains in version history');
  assert.ok(durableEdit.parentRevisionIds.includes(priorEditRevision.id), 'the edit descends from the exact prior revision');
  assert.equal(durableEdit.createdBy, 'human:release-author');
  const durableEditOperation = restarted.operations.get(durableEdit.operationId)!;
  assert.equal(durableEditOperation.kind, 'revise');
  assert.equal(durableEditOperation.actorId, 'human:release-author');
  assert.deepEqual(durableEditOperation.outputRevisionIds, [durableEdit.id]);

  const derivation = [...restarted.derivations.values()].find(
    (candidate) => candidate.childChunkId === createdChunkId,
  );
  assert.ok(derivation, 'generated output retains a durable derivation');
  assert.equal(derivation.via, 'generate');
  assert.equal(derivation.sourceRevisionId, proposalBeforeAcceptance.basisRevisionIds[0]);
  assert.equal(derivation.operationId, durableProposal.resolution?.operationId);

  console.log('release runtime OK — mixed text edit, durable restart, provenance, and safe projection');
} finally {
  await runtime?.close();
  rmSync(sandbox, { recursive: true, force: true });
}
