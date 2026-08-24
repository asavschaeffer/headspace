import assert from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHeadspaceHost } from '../src/host/api';
import { makeBlob } from '../src/kernel/hash';
import {
  newCommitId,
  newOperationId,
  newRevisionId,
} from '../src/kernel/ids';
import { deserializeState, type SerializedState } from '../src/kernel/serialize';
import {
  applyCommit,
  currentRevision,
  emptyState,
  InvariantError,
  materialize,
  revisionText,
} from '../src/kernel/state';
import {
  acceptProposal,
  createChunk,
  propose,
  rejectProposal,
  severOccurrence,
  staleReason,
  type TxCtx,
} from '../src/kernel/tx';
import type { Commit, Proposal } from '../src/kernel/types';

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
let tick = 0;
const now = () => new Date(Date.UTC(2026, 7, 22, 12, 0, 0, tick++)).toISOString();

function refuses(state: ReturnType<typeof emptyState>, commit: Commit, pattern: RegExp): void {
  const head = state.head;
  const count = state.commitCount;
  assert.throws(
    () => applyCommit(state, commit),
    (error: unknown) => error instanceof InvariantError && pattern.test(String(error)),
  );
  assert.equal(state.head, head);
  assert.equal(state.commitCount, count);
}

// A reusable truth base with two owners and two containers exercises every
// proposal reference kind without relying on private kernel internals.
const baseLog: Commit[] = [];
const baseState = emptyState();
const baseCtx: TxCtx = { state: baseState, actorId: 'human:test', now, onCommit: (commit) => baseLog.push(commit) };
const target = await createChunk(baseCtx, { text: 'reviewed target' });
const other = await createChunk(baseCtx, { text: 'other owner' });
const containerA = await createChunk(baseCtx, { text: 'container A' });
const child = await createChunk(baseCtx, { text: 'anchor', containerId: containerA.chunkId });
const containerB = await createChunk(baseCtx, { text: 'container B' });
const anchorId = child.occurrenceId!;

const proposalCommit = (): Commit => {
  const state = materialize(baseLog);
  return propose(
    { state, actorId: 'human:test', now },
    {
      kind: 'suggested-edit',
      basisRevisionIds: [target.revisionId],
      targetChunkIds: [target.chunkId],
      payload: [{ op: 'revise', chunkId: target.chunkId, text: 'reviewed replacement' }],
    },
  ).commit;
};

const mutateProposal = (
  mutate: (proposal: Proposal, commit: Commit) => void,
  pattern: RegExp,
): void => {
  const commit = clone(proposalCommit());
  mutate(commit.facts.proposals![0], commit);
  refuses(materialize(baseLog), commit, pattern);
};

mutateProposal((proposal) => {
  delete (proposal as { operationId?: string }).operationId;
}, /operationId must be a non-empty string/);
mutateProposal((proposal) => {
  proposal.producer = { id: 'forged.provider', version: '9' };
}, /producer disagrees/);
mutateProposal((proposal) => { proposal.status = 'accepted'; }, /must be open/);
mutateProposal((proposal, commit) => {
  proposal.basisRevisionIds = [other.revisionId];
  proposal.payload = [{ op: 'revise', chunkId: target.chunkId, text: 'wrong basis' }];
  commit.operation.inputRevisionIds = [other.revisionId];
}, /basis has no revision of chunk/);
mutateProposal((proposal) => {
  proposal.targetChunkIds = ['ch_missing'];
}, /unknown target chunk/);
mutateProposal((proposal) => {
  proposal.freshnessRevisionIds = [other.revisionId];
}, /freshness revision .* missing from operation inputs/);
mutateProposal((proposal, commit) => {
  proposal.freshnessRevisionIds = [target.revisionId];
  proposal.freshnessRevisionStates = [{
    chunkId: other.chunkId,
    revisionId: target.revisionId,
    followsCurrent: true,
    redacted: false,
    chunkTombstoned: false,
  }];
  commit.operation.inputRevisionIds = [target.revisionId];
}, /belongs to/);
mutateProposal((proposal) => {
  proposal.payload = [
    { op: 'create', tempId: 'same', text: 'one' },
    { op: 'create', tempId: 'same', text: 'two' },
  ];
}, /duplicate tempId/);
mutateProposal((proposal) => {
  proposal.payload = [{
    op: 'create',
    tempId: 'span',
    text: 'derived',
    derivedFrom: {
      sourceRevisionId: target.revisionId,
      via: 'extract',
      sourceSpan: { revisionId: target.revisionId, method: 'raw@1', start: 0, end: 100_000 },
    },
  }];
}, /ends past revision/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'place', containerId: containerA.chunkId, chunkId: { tempId: 'missing' } }];
}, /unknown tempId/);
mutateProposal((proposal) => {
  proposal.payload = [
    { op: 'place', containerId: containerA.chunkId, chunkId: { tempId: 'later' } },
    { op: 'create', tempId: 'later', text: 'too late' },
  ];
}, /before it is created/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'repin', occurrenceId: anchorId, revisionId: other.revisionId }];
}, /belongs to/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'place', containerId: containerB.chunkId, chunkId: target.chunkId, after: anchorId }];
}, /anchor .* belongs to/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'relate', fromChunkId: target.chunkId, role: 'references' }];
}, /exactly one destination/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'sever', occurrenceId: 'occ_missing' }];
}, /severs unknown occurrence/);
mutateProposal((proposal) => {
  proposal.freshnessStructure = {
    containers: [{ containerId: 'ch_missing', occurrences: [] }],
    placements: [],
  };
}, /structure names unknown container/);
mutateProposal((proposal) => {
  proposal.payload = [{ op: 'execute-now' } as never];
}, /unknown change op/);

// Within-commit duplicates are rejected before Map.set can collapse them.
{
  const commit = clone(proposalCommit());
  commit.facts.proposals!.push(clone(commit.facts.proposals![0]));
  refuses(materialize(baseLog), commit, /duplicated within commit/);
}

// Exact overwrite exploit: a current-parent propose commit may not reuse a
// durable ProposalId to replace its payload/author/producer in the Map fold.
const historyLog = [...baseLog];
const historyState = materialize(historyLog);
const original = propose(
  { state: historyState, actorId: 'human:reviewer', now, onCommit: (commit) => historyLog.push(commit) },
  {
    kind: 'suggested-edit',
    basisRevisionIds: [target.revisionId],
    targetChunkIds: [target.chunkId],
    payload: [{ op: 'revise', chunkId: target.chunkId, text: 'the reviewed text' }],
  },
);
{
  const operationId = newOperationId();
  const at = now();
  const overwritten: Proposal = {
    ...clone(historyState.proposals.get(original.proposalId)!),
    operationId,
    payload: [{ op: 'revise', chunkId: target.chunkId, text: 'silent overwrite' }],
    createdBy: 'agent:mallory',
  };
  const forged: Commit = {
    id: newCommitId(),
    parentIds: [historyState.head!],
    at,
    actorId: 'human:mallory',
    operation: {
      id: operationId,
      kind: 'propose',
      actorId: 'human:mallory',
      at,
      inputRevisionIds: [target.revisionId],
      outputRevisionIds: [],
      params: { kind: overwritten.kind },
    },
    facts: { proposals: [overwritten] },
  };
  refuses(historyState, forged, /proposal .* already exists/);
  assert.equal(historyState.proposals.get(original.proposalId)!.payload[0].op, 'revise');
  assert.equal((historyState.proposals.get(original.proposalId)!.payload[0] as { text: string }).text, 'the reviewed text');
}

// Resolution is a one-way, one-per-commit transition attributed to this exact
// terminal operation.
{
  const builder = materialize(historyLog);
  const valid = rejectProposal(
    { state: builder, actorId: 'human:reviewer', now },
    { proposalId: original.proposalId },
  );
  const duplicate = clone(valid);
  duplicate.facts.proposalUpdates!.push(clone(duplicate.facts.proposalUpdates![0]));
  refuses(materialize(historyLog), duplicate, /duplicate update/);

  const reopened = clone(valid);
  reopened.facts.proposalUpdates![0].status = 'open';
  refuses(materialize(historyLog), reopened, /cannot transition from open to open/);

  const wrongOperation = clone(valid);
  wrongOperation.facts.proposalUpdates![0].resolution!.operationId = baseLog[0].operation.id;
  refuses(materialize(historyLog), wrongOperation, /belongs to operation/);

  const missingResolutionOperation = clone(valid);
  delete (missingResolutionOperation.facts.proposalUpdates![0].resolution as { operationId?: string }).operationId;
  refuses(materialize(historyLog), missingResolutionOperation, /operationId must be a non-empty string/);

  const unresolved = clone(valid);
  delete unresolved.facts.proposalUpdates![0].resolution;
  refuses(materialize(historyLog), unresolved, /requires a resolution/);

  const terminalState = materialize(historyLog);
  applyCommit(terminalState, valid);
  const second = clone(valid);
  second.id = newCommitId();
  second.parentIds = [terminalState.head!];
  second.operation.id = newOperationId();
  second.facts.proposalUpdates![0].resolution!.operationId = second.operation.id;
  refuses(terminalState, second, /already rejected/);
}

// Operation envelope and attribution are append-once and internally exact.
{
  const first = baseLog[0];
  const adjacent = clone(first);
  adjacent.id = newCommitId();
  adjacent.parentIds = [first.id];
  adjacent.facts = {};
  adjacent.operation.outputRevisionIds = [];
  assert.throws(() => materialize([first, adjacent]), /operation .* already exists/);

  const state = materialize(baseLog);
  const at = now();
  const envelope = (operationId = newOperationId()): Commit => ({
    id: newCommitId(),
    parentIds: [state.head!],
    at,
    actorId: 'human:test',
    operation: {
      id: operationId,
      kind: 'revise',
      actorId: 'human:test',
      at,
      inputRevisionIds: [],
      outputRevisionIds: [],
    },
    facts: {},
  });
  const actorMismatch = envelope();
  actorMismatch.operation.actorId = 'human:mallory';
  refuses(state, actorMismatch, /actor does not match/);
  const unknownKind = envelope();
  (unknownKind.operation as { kind: string }).kind = 'execute';
  refuses(state, unknownKind, /unknown kind/);
  const missingEnvelope = envelope();
  (missingEnvelope as { actorId?: string }).actorId = undefined;
  (missingEnvelope.operation as { actorId?: string }).actorId = undefined;
  refuses(state, missingEnvelope, /commit\.actorId must be a non-empty string/);
  const unknownInput = envelope();
  unknownInput.operation.inputRevisionIds = ['rev_missing'];
  refuses(state, unknownInput, /input names unknown revision/);
  const wrongOutput = envelope();
  wrongOutput.operation.outputRevisionIds = [target.revisionId];
  refuses(state, wrongOutput, /outputs do not exactly match/);

  const oldAttribution = envelope();
  const blob = await makeBlob('text/plain', 'old operation attribution');
  const revisionId = newRevisionId();
  oldAttribution.facts = {
    blobs: [blob],
    revisions: [{
      id: revisionId,
      chunkId: target.chunkId,
      blobHash: blob.hash,
      mediaType: 'text/plain',
      parentRevisionIds: [target.revisionId],
      createdBy: 'human:test',
      createdAt: at,
      operationId: baseLog[0].operation.id,
    }],
    setCurrent: [{ chunkId: target.chunkId, revisionId }],
  };
  oldAttribution.operation.inputRevisionIds = [target.revisionId];
  oldAttribution.operation.outputRevisionIds = [revisionId];
  refuses(state, oldAttribution, /belongs to operation/);
}

// Propose and reject are inert operation kinds: even otherwise-valid revision
// facts cannot ride beside the proposal transition.
{
  const state = materialize(baseLog);
  const direct = clone(proposalCommit());
  const blob = await makeBlob('text/plain', 'truth before acceptance');
  const revisionId = newRevisionId();
  direct.facts.blobs = [blob];
  direct.facts.revisions = [{
    id: revisionId,
    chunkId: target.chunkId,
    blobHash: blob.hash,
    mediaType: 'text/plain',
    parentRevisionIds: [target.revisionId],
    createdBy: direct.operation.actorId,
    createdAt: direct.operation.at,
    operationId: direct.operation.id,
  }];
  direct.facts.setCurrent = [{ chunkId: target.chunkId, revisionId }];
  direct.operation.outputRevisionIds = [revisionId];
  refuses(state, direct, /propose operation cannot carry blobs facts/);
  assert.equal(revisionText(state, target.revisionId), 'reviewed target');

  const openState = materialize(historyLog);
  const rejectingState = materialize(historyLog);
  const rejected = rejectProposal({ state: rejectingState, actorId: 'human:test', now }, { proposalId: original.proposalId });
  const rejectBlob = await makeBlob('text/plain', 'truth during rejection');
  const rejectRevisionId = newRevisionId();
  const forgedReject = clone(rejected);
  forgedReject.facts.blobs = [rejectBlob];
  forgedReject.facts.revisions = [{
    id: rejectRevisionId,
    chunkId: target.chunkId,
    blobHash: rejectBlob.hash,
    mediaType: 'text/plain',
    parentRevisionIds: [target.revisionId],
    createdBy: forgedReject.operation.actorId,
    createdAt: forgedReject.operation.at,
    operationId: forgedReject.operation.id,
  }];
  forgedReject.facts.setCurrent = [{ chunkId: target.chunkId, revisionId: rejectRevisionId }];
  forgedReject.operation.outputRevisionIds = [rejectRevisionId];
  refuses(openState, forgedReject, /reject operation cannot carry blobs facts/);
}

// Acceptance is not a status convention: its exact facts must realize the
// reviewed text and structure. Status-only and different-text commits fail.
{
  const log: Commit[] = [];
  const state = emptyState();
  const ctx: TxCtx = { state, actorId: 'human:author', now, onCommit: (commit) => log.push(commit) };
  const doc = await createChunk(ctx, { text: 'before' });
  const suggestion = propose(ctx, {
    kind: 'suggested-edit',
    basisRevisionIds: [doc.revisionId],
    targetChunkIds: [doc.chunkId],
    payload: [{ op: 'revise', chunkId: doc.chunkId, text: 'reviewed after' }],
    createdBy: 'agent:writer',
  });
  const builderState = materialize(log);
  const valid = (await acceptProposal(
    { state: builderState, actorId: 'human:reviewer', now },
    { proposalId: suggestion.proposalId },
  )).commit;

  const statusOnly = clone(valid);
  statusOnly.facts = { proposalUpdates: clone(valid.facts.proposalUpdates) };
  statusOnly.operation.outputRevisionIds = [];
  refuses(materialize(log), statusOnly, /has 0 blobs fact\(s\), expected 1/);

  const differentText = clone(valid);
  const maliciousBlob = await makeBlob('text/plain', 'different valid text');
  differentText.facts.blobs![0] = maliciousBlob;
  differentText.facts.revisions![0].blobHash = maliciousBlob.hash;
  refuses(materialize(log), differentText, /does not exactly realize/);

  const admitted = materialize(log);
  applyCommit(admitted, valid);
  assert.equal(admitted.proposals.get(suggestion.proposalId)!.status, 'accepted');
  assert.equal(revisionText(admitted, currentRevision(admitted, doc.chunkId).id), 'reviewed after');
}

// Fresh IDs are one-to-one facts, not merely array slots: reusing one created
// chunk ID cannot collapse two reviewed creates into one Map entry.
{
  const log: Commit[] = [];
  const state = emptyState();
  const ctx: TxCtx = { state, actorId: 'human:author', now, onCommit: (commit) => log.push(commit) };
  const seed = await createChunk(ctx, { text: 'two creates' });
  const candidate = propose(ctx, {
    kind: 'generation',
    basisRevisionIds: [seed.revisionId],
    targetChunkIds: [seed.chunkId],
    payload: [
      { op: 'create', tempId: 'one', text: 'one' },
      { op: 'create', tempId: 'two', text: 'two' },
    ],
  });
  const builder = materialize(log);
  const accepted = (await acceptProposal(
    { state: builder, actorId: 'human:reviewer', now },
    { proposalId: candidate.proposalId },
  )).commit;
  const collapsed = clone(accepted);
  collapsed.facts.chunks![1].id = collapsed.facts.chunks![0].id;
  collapsed.facts.revisions![1].chunkId = collapsed.facts.chunks![0].id;
  refuses(materialize(log), collapsed, /chunk .* duplicated within commit/);
}

// A removed placement anchor turns the proposal stale and sedimentary; it does
// not make validation or acceptance throw halfway through.
{
  const state = materialize(baseLog);
  const placed = propose(
    { state, actorId: 'human:test', now },
    {
      kind: 'generation',
      basisRevisionIds: [containerA.revisionId],
      targetChunkIds: [containerA.chunkId],
      payload: [
        { op: 'create', tempId: 'new', text: 'after anchor' },
        { op: 'place', containerId: containerA.chunkId, chunkId: { tempId: 'new' }, after: anchorId },
      ],
    },
  );
  severOccurrence({ state, actorId: 'human:test', now }, { occurrenceId: anchorId });
  assert.match(staleReason(state, state.proposals.get(placed.proposalId)!) ?? '', /anchor occurrence .* no longer exists/);
  const result = await acceptProposal({ state, actorId: 'human:reviewer', now }, { proposalId: placed.proposalId });
  assert.equal(result.applied, false);
  assert.equal(state.proposals.get(placed.proposalId)!.status, 'superseded');
}

// Exercise the same duplicate-ID exploit through the live HTTP boundary.
{
  const root = mkdtempSync(join(tmpdir(), 'headspace-proposal-boundary-'));
  type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => void;
  const middlewares: Middleware[] = [];
  const server = createServer((req, res) => {
    let index = 0;
    const next = (): void => {
      const middleware = middlewares[index++];
      if (middleware) return middleware(req, res, next);
      res.statusCode = 404;
      res.end('not found');
    };
    next();
  });
  const runtime = createHeadspaceHost({ root, contentDirs: [], contentFiles: [], collaborators: [] });
  if (typeof runtime.plugin.configureServer !== 'function') throw new Error('configureServer hook missing');
  (runtime.plugin.configureServer as unknown as (server: {
    httpServer: unknown;
    middlewares: { use(middleware: Middleware): void };
  }) => unknown)({
    httpServer: server,
    middlewares: { use: (middleware: Middleware) => middlewares.push(middleware) },
  });
  try {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const { port } = server.address() as AddressInfo;
    const base = `http://127.0.0.1:${port}`;
    const payload = await (await fetch(`${base}/api/state`)).json() as { state: SerializedState };
    const clientState = deserializeState(payload.state);
    const seed = await createChunk({ state: clientState, actorId: 'human:client', now }, { text: 'API seed' });
    let response = await fetch(`${base}/api/commits`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ commits: [seed.commit] }),
    });
    assert.equal(response.status, 200);
    const suggestion = propose(
      { state: clientState, actorId: 'human:client', now },
      {
        kind: 'suggested-edit',
        basisRevisionIds: [seed.revisionId],
        targetChunkIds: [seed.chunkId],
        payload: [{ op: 'revise', chunkId: seed.chunkId, text: 'API reviewed' }],
      },
    );
    response = await fetch(`${base}/api/commits`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ commits: [suggestion.commit] }),
    });
    assert.equal(response.status, 200);

    const operationId = newOperationId();
    const at = now();
    const durable = clientState.proposals.get(suggestion.proposalId)!;
    const overwrite: Commit = {
      id: newCommitId(),
      parentIds: [clientState.head!],
      at,
      actorId: 'human:mallory',
      operation: {
        id: operationId,
        kind: 'propose',
        actorId: 'human:mallory',
        at,
        inputRevisionIds: [seed.revisionId],
        outputRevisionIds: [],
        params: { kind: durable.kind },
      },
      facts: {
        proposals: [{
          ...clone(durable),
          operationId,
          payload: [{ op: 'revise', chunkId: seed.chunkId, text: 'API overwrite' }],
        }],
      },
    };
    response = await fetch(`${base}/api/commits`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ commits: [overwrite] }),
    });
    assert.equal(response.status, 409);
    assert.match((await response.json() as { error: string }).error, /proposal .* already exists/);

    const after = await (await fetch(`${base}/api/state`)).json() as { state: SerializedState };
    const durableAfter = after.state.proposals.find((proposal) => proposal.id === suggestion.proposalId)!;
    assert.equal((durableAfter.payload[0] as { text: string }).text, 'API reviewed');
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    runtime.close();
    rmSync(root, { recursive: true, force: true });
  }
}

console.log('proposal invariants OK — immutable history, inert review, exact acceptance, and strict current-format admission');
