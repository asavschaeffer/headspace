import type {
  BlobHash,
  Chunk,
  ChunkId,
  Commit,
  Occurrence,
  OccurrenceId,
  Proposal,
  ProposedChange,
  Revision,
  RevisionId,
  SpanAddress,
} from './types';
import { isCompositeMediaType, MEDIA_TEXT } from './types';
import { keyBetween } from './fractional';

// The materialized truth: what the log says, folded into queryable maps.
// The log (commits) is authoritative; this state is its deterministic product.

export interface WorkspaceGraph {
  chunks: Map<ChunkId, Chunk>;
  revisions: Map<RevisionId, Revision>;
  blobs: Map<BlobHash, import('./types').Blob>;
  occurrences: Map<OccurrenceId, Occurrence>;
  links: Map<string, import('./types').Link>;
  derivations: Map<string, import('./types').Derivation>;
  proposals: Map<string, Proposal>;
  operations: Map<string, import('./types').Operation>;
  head: string | null; // last applied commit id
  commitCount: number;
}

export function emptyState(): WorkspaceGraph {
  return {
    chunks: new Map(),
    revisions: new Map(),
    blobs: new Map(),
    occurrences: new Map(),
    links: new Map(),
    derivations: new Map(),
    proposals: new Map(),
    operations: new Map(),
    head: null,
    commitCount: 0,
  };
}

export class InvariantError extends Error {}

function fail(msg: string): never {
  throw new InvariantError(msg);
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const requireNonEmptyString = (value: unknown, label: string): string => {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} must be a non-empty string`);
  return value;
};

function requireUniqueStrings(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) fail(`${label} must be an array`);
  const seen = new Set<string>();
  for (const item of value) {
    const id = requireNonEmptyString(item, `${label} entry`);
    if (seen.has(id)) fail(`${label} contains duplicate ${id}`);
    seen.add(id);
  }
  return [...seen];
}

function validateSpan(
  span: SpanAddress,
  label: string,
  revisionOf: (id: RevisionId) => Revision | undefined,
  blobTextOf: (revision: Revision) => string | undefined,
  expectedRevisionId?: RevisionId,
): void {
  if (!isRecord(span)) fail(`${label} must be a span address`);
  const revisionId = requireNonEmptyString(span.revisionId, `${label}.revisionId`);
  const revision = revisionOf(revisionId);
  if (!revision) fail(`${label} addresses unknown revision ${revisionId}`);
  if (expectedRevisionId !== undefined && revisionId !== expectedRevisionId) {
    fail(`${label} addresses ${revisionId}, not source revision ${expectedRevisionId}`);
  }
  requireNonEmptyString(span.method, `${label}.method`);
  if (!Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end) || span.start < 0 || span.end < span.start) {
    fail(`${label} has invalid offsets ${span.start}:${span.end}`);
  }
  const text = blobTextOf(revision);
  if (text === undefined) fail(`${label} revision ${revisionId} has no readable blob`);
  if (span.end > text.length) fail(`${label} ends past revision ${revisionId}`);
}

// Would placing `chunkId` inside `containerId` close a containment cycle?
// Placement adds the edge container -> chunk; a cycle exists iff container is
// reachable downward from chunk (or they are the same identity). `extraEdges`
// lets a commit be validated with its own not-yet-applied occurrences included.
export function wouldCreateCycle(
  state: WorkspaceGraph,
  containerId: ChunkId,
  chunkId: ChunkId,
  extraEdges: { containerId: ChunkId; chunkId: ChunkId }[] = [],
): boolean {
  if (containerId === chunkId) return true;
  const seen = new Set<ChunkId>();
  const stack = [chunkId];
  while (stack.length) {
    const cur = stack.pop()!;
    if (cur === containerId) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const occ of state.occurrences.values()) {
      if (occ.containerId === cur) stack.push(occ.chunkId);
    }
    for (const e of extraEdges) {
      if (e.containerId === cur) stack.push(e.chunkId);
    }
  }
  return false;
}

const FACT_KEYS: Array<keyof Commit['facts']> = [
  'blobs',
  'chunks',
  'revisions',
  'setCurrent',
  'occurrences',
  'occurrenceUpdates',
  'removeOccurrences',
  'links',
  'removeLinks',
  'derivations',
  'proposals',
  'proposalUpdates',
  'tombstone',
  'redactRevisions',
];

function requireOnlyFactKinds(commit: Commit, allowed: ReadonlySet<keyof Commit['facts']>): void {
  for (const key of FACT_KEYS) {
    if (allowed.has(key)) continue;
    const value = commit.facts[key];
    if (value !== undefined && (!Array.isArray(value) || value.length > 0)) {
      fail(`${commit.operation.kind} operation cannot carry ${key} facts`);
    }
  }
}

const sameStrings = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index]);

const sameSpan = (left: SpanAddress | undefined, right: SpanAddress | undefined): boolean =>
  left === undefined
    ? right === undefined
    : right !== undefined &&
      left.revisionId === right.revisionId &&
      left.method === right.method &&
      left.start === right.start &&
      left.end === right.end;

const sameExternal = (
  left: import('./types').ExternalRef | undefined,
  right: import('./types').ExternalRef | undefined,
): boolean =>
  left === undefined
    ? right === undefined
    : right !== undefined &&
      left.layer === right.layer &&
      left.key === right.key &&
      left.url === right.url &&
      left.snapshotAt === right.snapshotAt;

const sameProducer = (
  left: import('./types').ProducerRef | undefined,
  right: unknown,
): boolean =>
  left === undefined
    ? right === undefined
    : isRecord(right) &&
      left.id === right.id &&
      left.version === right.version &&
      left.implementation === right.implementation &&
      left.receiptId === right.receiptId;

// Prove that an accept operation is the exact deterministic realization of
// the reviewed payload. IDs may be freshly minted, but every semantic field,
// array cardinality, ordering decision, and authorship edge is fixed.
function validateAcceptedRealization(state: WorkspaceGraph, commit: Commit, proposal: Proposal): void {
  const stale = proposalStaleReason(state, proposal);
  if (stale) fail(`accept operation cannot apply stale proposal ${proposal.id}: ${stale}`);

  const params = commit.operation.params;
  if (!isRecord(params) || params.proposalId !== proposal.id || params.proposalKind !== proposal.kind) {
    fail(`accept operation is not bound to proposal ${proposal.id} and kind ${proposal.kind}`);
  }
  const inputs = requireUniqueStrings(commit.operation.inputRevisionIds, 'accept inputRevisionIds');
  const basis = requireUniqueStrings(proposal.basisRevisionIds, `proposal ${proposal.id} basisRevisionIds`);
  if (inputs.length !== basis.length || inputs.some((id) => !basis.includes(id))) {
    fail(`accept operation inputs do not exactly match proposal ${proposal.id} basis`);
  }

  requireOnlyFactKinds(commit, new Set([
    'blobs',
    'chunks',
    'revisions',
    'setCurrent',
    'occurrences',
    'occurrenceUpdates',
    'removeOccurrences',
    'links',
    'derivations',
    'proposalUpdates',
  ]));
  const blobs = commit.facts.blobs ?? [];
  const chunks = commit.facts.chunks ?? [];
  const revisions = commit.facts.revisions ?? [];
  const setCurrent = commit.facts.setCurrent ?? [];
  const occurrences = commit.facts.occurrences ?? [];
  const occurrenceUpdates = commit.facts.occurrenceUpdates ?? [];
  const removeOccurrences = commit.facts.removeOccurrences ?? [];
  const links = commit.facts.links ?? [];
  const derivations = commit.facts.derivations ?? [];

  const createCount = proposal.payload.filter((change) => change.op === 'create').length;
  const reviseCount = proposal.payload.filter((change) => change.op === 'revise').length;
  const placeCount = proposal.payload.filter((change) => change.op === 'place').length;
  const repinCount = proposal.payload.filter((change) => change.op === 'repin').length;
  const relateCount = proposal.payload.filter((change) => change.op === 'relate').length;
  const severCount = proposal.payload.filter((change) => change.op === 'sever').length;
  const derivationCount = proposal.payload.filter(
    (change) => change.op === 'create' && change.derivedFrom !== undefined,
  ).length;
  const expectedCounts: Array<[string, number, number]> = [
    ['blobs', blobs.length, createCount + reviseCount],
    ['chunks', chunks.length, createCount],
    ['revisions', revisions.length, createCount + reviseCount],
    ['setCurrent', setCurrent.length, reviseCount],
    ['occurrences', occurrences.length, placeCount],
    ['occurrenceUpdates', occurrenceUpdates.length, repinCount],
    ['removeOccurrences', removeOccurrences.length, severCount],
    ['links', links.length, relateCount],
    ['derivations', derivations.length, derivationCount],
  ];
  for (const [label, actual, expected] of expectedCounts) {
    if (actual !== expected) {
      fail(`accept operation for proposal ${proposal.id} has ${actual} ${label} fact(s), expected ${expected}`);
    }
  }

  let blobIndex = 0;
  let chunkIndex = 0;
  let revisionIndex = 0;
  let currentIndex = 0;
  let occurrenceIndex = 0;
  let occurrenceUpdateIndex = 0;
  let removeOccurrenceIndex = 0;
  let linkIndex = 0;
  let derivationIndex = 0;
  const tempChunks = new Map<string, ChunkId>();
  const cursor = new Map<ChunkId, { last: string | null; bound: string | null }>();
  const nextPosition = (containerId: ChunkId, after?: OccurrenceId, at?: 'start'): string => {
    let placement: { last: string | null; bound: string | null };
    if (after !== undefined) {
      const siblings = childOccurrences(state, containerId);
      const index = siblings.findIndex((occurrence) => occurrence.id === after);
      if (index < 0) fail(`accept operation anchor ${after} is missing from ${containerId}`);
      placement = { last: siblings[index].position, bound: siblings[index + 1]?.position ?? null };
    } else if (at === 'start') {
      const siblings = childOccurrences(state, containerId);
      placement = { last: null, bound: siblings[0]?.position ?? null };
    } else {
      placement = cursor.get(containerId) ?? (() => {
        const siblings = childOccurrences(state, containerId);
        return { last: siblings.at(-1)?.position ?? null, bound: null };
      })();
    }
    const position = keyBetween(placement.last, placement.bound);
    cursor.set(containerId, { last: position, bound: placement.bound });
    return position;
  };
  const validateContentRevision = (
    revision: Revision,
    blob: import('./types').Blob,
    expected: { chunkId: ChunkId; text: string; mediaType: string; parents: RevisionId[] },
  ): void => {
    if (
      revision.chunkId !== expected.chunkId ||
      revision.blobHash !== blob.hash ||
      revision.mediaType !== expected.mediaType ||
      !sameStrings(revision.parentRevisionIds, expected.parents) ||
      revision.createdBy !== proposal.createdBy ||
      revision.createdAt !== commit.operation.at ||
      revision.operationId !== commit.operation.id ||
      Boolean(revision.redacted) ||
      blob.mediaType !== expected.mediaType ||
      blob.text !== expected.text
    ) {
      fail(`accept operation revision ${revision.id} does not exactly realize proposal ${proposal.id}`);
    }
  };

  for (const change of proposal.payload) {
    if (change.op === 'create') {
      const chunk = chunks[chunkIndex++];
      const revision = revisions[revisionIndex++];
      const blob = blobs[blobIndex++];
      const mediaType = change.mediaType ?? MEDIA_TEXT;
      if (!chunk || !revision || !blob) fail(`accept operation is missing create facts for ${change.tempId}`);
      if (tempChunks.has(change.tempId)) fail(`accept operation repeats tempId ${change.tempId}`);
      tempChunks.set(change.tempId, chunk.id);
      if (chunk.currentRevisionId !== revision.id || chunk.tombstoned) {
        fail(`accept operation chunk ${chunk.id} does not realize create ${change.tempId}`);
      }
      validateContentRevision(revision, blob, {
        chunkId: chunk.id,
        text: change.text,
        mediaType,
        parents: [],
      });
      if (change.derivedFrom !== undefined) {
        const derivation = derivations[derivationIndex++];
        if (
          !derivation ||
          derivation.childChunkId !== chunk.id ||
          derivation.sourceRevisionId !== change.derivedFrom.sourceRevisionId ||
          derivation.via !== change.derivedFrom.via ||
          !sameSpan(derivation.sourceSpan, change.derivedFrom.sourceSpan) ||
          derivation.operationId !== commit.operation.id
        ) {
          fail(`accept operation derivation for ${change.tempId} does not exactly realize proposal ${proposal.id}`);
        }
      }
    } else if (change.op === 'revise') {
      const current = currentRevision(state, change.chunkId);
      const revision = revisions[revisionIndex++];
      const blob = blobs[blobIndex++];
      const pointer = setCurrent[currentIndex++];
      if (!revision || !blob || !pointer) fail(`accept operation is missing revise facts for ${change.chunkId}`);
      validateContentRevision(revision, blob, {
        chunkId: change.chunkId,
        text: change.text,
        mediaType: change.mediaType ?? current.mediaType,
        parents: change.mergeParentRevisionIds ?? [current.id],
      });
      if (pointer.chunkId !== change.chunkId || pointer.revisionId !== revision.id) {
        fail(`accept operation current pointer does not realize revision of ${change.chunkId}`);
      }
    } else if (change.op === 'place') {
      const occurrence = occurrences[occurrenceIndex++];
      const chunkId = typeof change.chunkId === 'string'
        ? change.chunkId
        : tempChunks.get(change.chunkId.tempId);
      if (!occurrence || !chunkId) fail(`accept operation cannot resolve proposed placement`);
      const expectedPosition = nextPosition(change.containerId, change.after, change.at);
      if (
        occurrence.containerId !== change.containerId ||
        occurrence.chunkId !== chunkId ||
        occurrence.position !== expectedPosition ||
        occurrence.mode !== (change.mode ?? 'contain') ||
        occurrence.pin !== 'current' ||
        occurrence.watch !== (change.watch ?? false)
      ) {
        fail(`accept operation occurrence ${occurrence.id} does not exactly realize proposal ${proposal.id}`);
      }
    } else if (change.op === 'repin') {
      const update = occurrenceUpdates[occurrenceUpdateIndex++];
      if (
        !update ||
        update.id !== change.occurrenceId ||
        update.pin !== change.revisionId ||
        update.position !== undefined ||
        update.watch !== undefined
      ) {
        fail(`accept operation repin does not exactly realize proposal ${proposal.id}`);
      }
    } else if (change.op === 'relate') {
      const link = links[linkIndex++];
      if (
        !link ||
        link.fromChunkId !== change.fromChunkId ||
        link.fromSpan !== undefined ||
        link.toChunkId !== change.toChunkId ||
        link.toRevisionId !== undefined ||
        !sameExternal(link.toExternal, change.toExternal) ||
        link.role !== change.role ||
        link.operationId !== commit.operation.id
      ) {
        fail(`accept operation relation does not exactly realize proposal ${proposal.id}`);
      }
    } else if (change.op === 'sever') {
      if (removeOccurrences[removeOccurrenceIndex++] !== change.occurrenceId) {
        fail(`accept operation sever does not exactly realize proposal ${proposal.id}`);
      }
    }
  }
}

// Validation is separable from folding so a caller can prove a commit is legal,
// make it durable, and only then advance memory — an append that throws must
// never leave state ahead of the log (store.md: one operation, one commit, one
// log append). Throws InvariantError; mutates nothing.
export function validateCommit(
  state: WorkspaceGraph,
  commit: Commit,
): void {
  const rawCommit: unknown = commit;
  if (!isRecord(rawCommit)) fail(`commit must be an object`);
  const rawOperation: unknown = rawCommit.operation;
  const rawFacts: unknown = rawCommit.facts;
  if (!isRecord(rawOperation)) fail(`commit operation must be an object`);
  if (!isRecord(rawFacts)) fail(`commit facts must be an object`);
  requireNonEmptyString(commit.id, 'commit.id');
  requireNonEmptyString(commit.actorId, 'commit.actorId');
  requireNonEmptyString(commit.at, 'commit.at');
  requireNonEmptyString(commit.operation.id, 'operation.id');
  requireNonEmptyString(commit.operation.actorId, 'operation.actorId');
  requireNonEmptyString(commit.operation.at, 'operation.at');
  requireUniqueStrings(commit.parentIds, 'commit.parentIds');
  const operationKinds = new Set([
    'create',
    'revise',
    'place',
    'move',
    'sever',
    'relate',
    'unrelate',
    'copy',
    'reference',
    'transclude',
    'promote',
    'propose',
    'accept',
    'reject',
    'tombstone',
    'redact',
    'import',
    'reconcile',
  ]);
  if (!operationKinds.has(commit.operation.kind)) {
    fail(`operation ${commit.operation.id} has unknown kind ${String(commit.operation.kind)}`);
  }
  const f = commit.facts;
  if (state.operations.has(commit.operation.id)) {
    fail(`operation ${commit.operation.id} already exists`);
  }
  if (commit.operation.actorId !== commit.actorId) {
    fail(`operation ${commit.operation.id} actor does not match commit actor`);
  }
  if (commit.operation.at !== commit.at) {
    fail(`operation ${commit.operation.id} time does not match commit time`);
  }
  const newChunks = new Set((f.chunks ?? []).map((c) => c.id));

  const blobsInCommit = new Map<string, import('./types').Blob>();
  for (const b of f.blobs ?? []) {
    const existing = state.blobs.get(b.hash);
    if (existing && (existing.text !== b.text || existing.mediaType !== b.mediaType)) {
      fail(`blob ${b.hash} already exists with different content; blobs are immutable`);
    }
    const earlier = blobsInCommit.get(b.hash);
    if (earlier && (earlier.text !== b.text || earlier.mediaType !== b.mediaType)) {
      fail(`blob ${b.hash} is duplicated within commit with different content`);
    }
    blobsInCommit.set(b.hash, b);
  }
  const revisionIdsInCommit = new Set<string>();
  for (const r of f.revisions ?? []) {
    if (state.revisions.has(r.id)) fail(`revision ${r.id} already exists; revisions are immutable`);
    if (revisionIdsInCommit.has(r.id)) fail(`revision ${r.id} is duplicated within commit`);
    revisionIdsInCommit.add(r.id);
    if (!state.chunks.has(r.chunkId) && !newChunks.has(r.chunkId)) fail(`revision ${r.id} for unknown chunk ${r.chunkId}`);
    const hasBlob = state.blobs.has(r.blobHash) || (f.blobs ?? []).some((b) => b.hash === r.blobHash);
    if (!hasBlob) fail(`revision ${r.id} points at missing blob ${r.blobHash}`);
    for (const p of r.parentRevisionIds) {
      if (!state.revisions.has(p)) fail(`revision ${r.id} has unknown parent ${p}`);
    }
    if (r.operationId !== commit.operation.id) {
      fail(`revision ${r.id} belongs to operation ${r.operationId}, not ${commit.operation.id}`);
    }
  }
  const newRevisions = new Map((f.revisions ?? []).map((r) => [r.id, r]));
  const chunkIdsInCommit = new Set<string>();
  for (const c of f.chunks ?? []) {
    if (state.chunks.has(c.id)) fail(`chunk ${c.id} already exists`);
    if (chunkIdsInCommit.has(c.id)) fail(`chunk ${c.id} is duplicated within commit`);
    chunkIdsInCommit.add(c.id);
    const rev = newRevisions.get(c.currentRevisionId);
    if (!rev || rev.chunkId !== c.id) fail(`chunk ${c.id} current revision must be its own new revision`);
  }
  const currentChunksInCommit = new Set<string>();
  for (const s of f.setCurrent ?? []) {
    if (currentChunksInCommit.has(s.chunkId)) fail(`setCurrent: duplicate update for chunk ${s.chunkId}`);
    currentChunksInCommit.add(s.chunkId);
    const rev = newRevisions.get(s.revisionId) ?? state.revisions.get(s.revisionId);
    if (!rev) fail(`setCurrent: unknown revision ${s.revisionId}`);
    if (rev!.chunkId !== s.chunkId) fail(`setCurrent: revision ${s.revisionId} belongs to ${rev!.chunkId}, not ${s.chunkId}`);
    if (!state.chunks.has(s.chunkId) && !newChunks.has(s.chunkId)) fail(`setCurrent: unknown chunk ${s.chunkId}`);
  }
  const newEdges = (f.occurrences ?? []).map((o) => ({ containerId: o.containerId, chunkId: o.chunkId }));
  const occurrenceIdsInCommit = new Set<string>();
  for (const [i, o] of (f.occurrences ?? []).entries()) {
    if (state.occurrences.has(o.id)) fail(`occurrence ${o.id} already exists`);
    if (occurrenceIdsInCommit.has(o.id)) fail(`occurrence ${o.id} is duplicated within commit`);
    occurrenceIdsInCommit.add(o.id);
    if (!state.chunks.has(o.containerId) && !newChunks.has(o.containerId)) fail(`occurrence container ${o.containerId} unknown`);
    if (!state.chunks.has(o.chunkId) && !newChunks.has(o.chunkId)) fail(`occurrence chunk ${o.chunkId} unknown`);
    if (!o.position) fail(`occurrence ${o.id} missing position`);
    if (wouldCreateCycle(state, o.containerId, o.chunkId, newEdges.slice(0, i))) {
      fail(`occurrence ${o.id} would create a containment cycle (${o.containerId} <- ${o.chunkId})`);
    }
  }
  const occurrenceUpdateIds = new Set<string>();
  for (const u of f.occurrenceUpdates ?? []) {
    if (occurrenceUpdateIds.has(u.id)) fail(`occurrenceUpdate: duplicate update for occurrence ${u.id}`);
    occurrenceUpdateIds.add(u.id);
    const occ = state.occurrences.get(u.id);
    if (!occ) fail(`occurrenceUpdate: unknown occurrence ${u.id}`);
    if (u.pin !== undefined && u.pin !== 'current') {
      const pinRev = state.revisions.get(u.pin) ?? newRevisions.get(u.pin);
      if (!pinRev) fail(`occurrenceUpdate: unknown pin revision ${u.pin}`);
      if (pinRev!.chunkId !== occ!.chunkId) {
        fail(`occurrenceUpdate: pin revision ${u.pin} belongs to ${pinRev!.chunkId}, not ${occ!.chunkId}`);
      }
    }
  }
  const removedOccurrenceIds = new Set<string>();
  for (const id of f.removeOccurrences ?? []) {
    if (removedOccurrenceIds.has(id)) fail(`removeOccurrences: duplicate occurrence ${id}`);
    removedOccurrenceIds.add(id);
    if (!state.occurrences.has(id)) fail(`removeOccurrences: unknown occurrence ${id}`);
  }
  // Links and derivations assert facts ABOUT other objects. A dangling one
  // survives replay forever and indexes as evidence pointing nowhere, so the
  // endpoints are checked here — the same bar occurrences already meet.
  const knownChunk = (id: ChunkId) => state.chunks.has(id) || newChunks.has(id);
  const knownRevision = (id: RevisionId) => state.revisions.has(id) || newRevisions.has(id);
  const linkIdsInCommit = new Set<string>();
  for (const l of f.links ?? []) {
    if (state.links.has(l.id)) fail(`link ${l.id} already exists`);
    if (linkIdsInCommit.has(l.id)) fail(`link ${l.id} is duplicated within commit`);
    linkIdsInCommit.add(l.id);
    if (!knownChunk(l.fromChunkId)) fail(`link ${l.id} starts at unknown chunk ${l.fromChunkId}`);
    if (l.toChunkId !== undefined && !knownChunk(l.toChunkId)) fail(`link ${l.id} points at unknown chunk ${l.toChunkId}`);
    if (l.toRevisionId !== undefined && !knownRevision(l.toRevisionId)) {
      fail(`link ${l.id} points at unknown revision ${l.toRevisionId}`);
    }
    if (l.fromSpan && !knownRevision(l.fromSpan.revisionId)) {
      fail(`link ${l.id} is anchored in unknown revision ${l.fromSpan.revisionId}`);
    }
    if (l.operationId !== commit.operation.id) {
      fail(`link ${l.id} belongs to operation ${l.operationId}, not ${commit.operation.id}`);
    }
  }
  const removedLinkIds = new Set<string>();
  for (const id of f.removeLinks ?? []) {
    if (removedLinkIds.has(id)) fail(`removeLinks: duplicate link ${id}`);
    removedLinkIds.add(id);
    if (!state.links.has(id)) fail(`removeLinks: unknown link ${id}`);
  }
  const derivationIdsInCommit = new Set<string>();
  for (const d of f.derivations ?? []) {
    if (state.derivations.has(d.id)) fail(`derivation ${d.id} already exists`);
    if (derivationIdsInCommit.has(d.id)) fail(`derivation ${d.id} is duplicated within commit`);
    derivationIdsInCommit.add(d.id);
    if (!knownChunk(d.childChunkId)) fail(`derivation ${d.id} names unknown child chunk ${d.childChunkId}`);
    if (!knownRevision(d.sourceRevisionId)) fail(`derivation ${d.id} names unknown source revision ${d.sourceRevisionId}`);
    if (d.sourceSpan && !knownRevision(d.sourceSpan.revisionId)) {
      fail(`derivation ${d.id} addresses unknown revision ${d.sourceSpan.revisionId}`);
    }
    if (d.operationId !== commit.operation.id) {
      fail(`derivation ${d.id} belongs to operation ${d.operationId}, not ${commit.operation.id}`);
    }
  }

  // Proposals are durable claims about a precise state, despite being inert
  // until accepted. Validate their complete shape at admission so replay can
  // never replace proposal history or turn a malformed payload into truth.
  const proposals = f.proposals ?? [];
  if (proposals.length > 0 && commit.operation.kind !== 'propose') {
    fail(`proposal facts require a propose operation, not ${commit.operation.kind}`);
  }
  const proposedIds = new Set<string>();
  for (const p of proposals) {
    const proposalId = requireNonEmptyString(p.id, 'proposal id');
    if (state.proposals.has(proposalId)) fail(`proposal ${proposalId} already exists`);
    if (proposedIds.has(proposalId)) fail(`proposal ${proposalId} is duplicated within commit`);
    proposedIds.add(proposalId);
  }

  const revisionOf = (id: RevisionId): Revision | undefined => newRevisions.get(id) ?? state.revisions.get(id);
  const blobTextOf = (revision: Revision): string | undefined =>
    state.blobs.get(revision.blobHash)?.text ?? (f.blobs ?? []).find((blob) => blob.hash === revision.blobHash)?.text;
  const proposalInputs = requireUniqueStrings(commit.operation.inputRevisionIds, 'operation inputRevisionIds');
  for (const revisionId of proposalInputs) {
    if (!knownRevision(revisionId)) fail(`operation input names unknown revision ${revisionId}`);
  }
  const inputSet = new Set(proposalInputs);
  const operationOutputs = requireUniqueStrings(commit.operation.outputRevisionIds, 'operation outputRevisionIds');
  const outputSet = new Set(operationOutputs);
  if (
    outputSet.size !== newRevisions.size ||
    [...newRevisions.keys()].some((revisionId) => !outputSet.has(revisionId))
  ) {
    fail(`operation ${commit.operation.id} outputs do not exactly match its new revisions`);
  }
  const requireInput = (proposalId: string, revisionId: RevisionId, kind: string): void => {
    if (!inputSet.has(revisionId)) {
      fail(`proposal ${proposalId} ${kind} revision ${revisionId} is missing from operation inputs`);
    }
  };
  const validateOccurrenceSnapshot = (
    proposalId: string,
    value: unknown,
    label: string,
  ): import('./types').OccurrencePrecondition => {
    if (!isRecord(value)) fail(`proposal ${proposalId} ${label} must be an occurrence precondition`);
    const occurrence = value as unknown as import('./types').OccurrencePrecondition;
    const id = requireNonEmptyString(occurrence.id, `proposal ${proposalId} ${label}.id`);
    const containerId = requireNonEmptyString(
      occurrence.containerId,
      `proposal ${proposalId} ${label}.containerId`,
    );
    const chunkId = requireNonEmptyString(occurrence.chunkId, `proposal ${proposalId} ${label}.chunkId`);
    if (!knownChunk(containerId)) fail(`proposal ${proposalId} ${label} names unknown container ${containerId}`);
    if (!knownChunk(chunkId)) fail(`proposal ${proposalId} ${label} names unknown chunk ${chunkId}`);
    requireNonEmptyString(occurrence.position, `proposal ${proposalId} ${label}.position`);
    if (occurrence.mode !== 'contain' && occurrence.mode !== 'transclude') {
      fail(`proposal ${proposalId} ${label} has invalid occurrence mode`);
    }
    if (typeof occurrence.watch !== 'boolean') fail(`proposal ${proposalId} ${label}.watch must be boolean`);
    if (occurrence.pin !== 'current') {
      const pin = requireNonEmptyString(occurrence.pin, `proposal ${proposalId} ${label}.pin`);
      const revision = revisionOf(pin);
      if (!revision) fail(`proposal ${proposalId} ${label} pins unknown revision ${pin}`);
      if (revision.chunkId !== chunkId) {
        fail(`proposal ${proposalId} ${label} pin ${pin} belongs to ${revision.chunkId}, not ${chunkId}`);
      }
    }
    return occurrence;
  };
  const occurrenceSnapshotEqual = (
    actual: Occurrence,
    expected: import('./types').OccurrencePrecondition,
  ): boolean =>
    actual.id === expected.id &&
    actual.containerId === expected.containerId &&
    actual.chunkId === expected.chunkId &&
    actual.position === expected.position &&
    actual.mode === expected.mode &&
    actual.pin === expected.pin &&
    actual.watch === expected.watch;

  const proposalKinds = new Set([
    'generation',
    'source-update',
    'suggested-edit',
    'detected-relation',
    'merge',
    'reconciliation',
  ]);
  for (const p of proposals) {
    const proposalId = p.id;
    if (!proposalKinds.has(p.kind)) fail(`proposal ${proposalId} has unknown kind ${String(p.kind)}`);
    if (p.status !== 'open') fail(`new proposal ${proposalId} must be open`);
    if (p.resolution !== undefined) fail(`new proposal ${proposalId} cannot already have a resolution`);
    requireNonEmptyString(p.operationId, `proposal ${proposalId}.operationId`);
    if (p.operationId !== commit.operation.id) {
      fail(`proposal ${proposalId} belongs to operation ${p.operationId}, not ${commit.operation.id}`);
    }
    requireNonEmptyString(p.createdBy, `proposal ${proposalId}.createdBy`);
    requireNonEmptyString(p.createdAt, `proposal ${proposalId}.createdAt`);
    if (p.createdAt !== commit.operation.at) {
      fail(`proposal ${proposalId} creation time does not match its propose operation`);
    }
    if (p.note !== undefined && typeof p.note !== 'string') {
      fail(`proposal ${proposalId}.note must be a string`);
    }

    const basisRevisionIds = requireUniqueStrings(p.basisRevisionIds, `proposal ${proposalId} basisRevisionIds`);
    for (const revisionId of basisRevisionIds) {
      if (!knownRevision(revisionId)) fail(`proposal ${proposalId} has unknown basis revision ${revisionId}`);
      requireInput(proposalId, revisionId, 'basis');
    }
    const basis = new Set(basisRevisionIds);

    const targetChunkIds = requireUniqueStrings(p.targetChunkIds, `proposal ${proposalId} targetChunkIds`);
    for (const chunkId of targetChunkIds) {
      if (!knownChunk(chunkId)) fail(`proposal ${proposalId} has unknown target chunk ${chunkId}`);
    }

    let freshnessRevisionIds: string[] | undefined;
    if (p.freshnessRevisionIds !== undefined) {
      freshnessRevisionIds = requireUniqueStrings(
        p.freshnessRevisionIds,
        `proposal ${proposalId} freshnessRevisionIds`,
      );
      for (const revisionId of freshnessRevisionIds) {
        if (!knownRevision(revisionId)) fail(`proposal ${proposalId} has unknown freshness revision ${revisionId}`);
        requireInput(proposalId, revisionId, 'freshness');
      }
    }

    if (p.freshnessRevisionStates !== undefined) {
      if (!Array.isArray(p.freshnessRevisionStates)) {
        fail(`proposal ${proposalId} freshnessRevisionStates must be an array`);
      }
      const stateRevisionIds = new Set<string>();
      const followsCurrentIds = new Set<string>();
      for (const expected of p.freshnessRevisionStates) {
        if (!isRecord(expected)) fail(`proposal ${proposalId} has an invalid freshness revision state`);
        const revisionId = requireNonEmptyString(
          expected.revisionId,
          `proposal ${proposalId} freshness state revisionId`,
        );
        const chunkId = requireNonEmptyString(expected.chunkId, `proposal ${proposalId} freshness state chunkId`);
        if (stateRevisionIds.has(revisionId)) {
          fail(`proposal ${proposalId} has duplicate freshness state for ${revisionId}`);
        }
        stateRevisionIds.add(revisionId);
        const revision = revisionOf(revisionId);
        if (!revision) fail(`proposal ${proposalId} has unknown freshness state revision ${revisionId}`);
        if (revision.chunkId !== chunkId) {
          fail(`proposal ${proposalId} freshness revision ${revisionId} belongs to ${revision.chunkId}, not ${chunkId}`);
        }
        const chunk = state.chunks.get(chunkId) ?? (f.chunks ?? []).find((candidate) => candidate.id === chunkId);
        if (!chunk) fail(`proposal ${proposalId} freshness state names unknown chunk ${chunkId}`);
        if (typeof expected.followsCurrent !== 'boolean' || typeof expected.redacted !== 'boolean' ||
          typeof expected.chunkTombstoned !== 'boolean') {
          fail(`proposal ${proposalId} freshness state ${revisionId} has invalid flags`);
        }
        if (expected.followsCurrent) {
          followsCurrentIds.add(revisionId);
        }
        requireInput(proposalId, revisionId, 'freshness-state');
      }
      if (freshnessRevisionIds !== undefined) {
        const declared = new Set(freshnessRevisionIds);
        if (
          declared.size !== followsCurrentIds.size ||
          [...declared].some((revisionId) => !followsCurrentIds.has(revisionId))
        ) {
          fail(`proposal ${proposalId} freshnessRevisionIds disagree with freshness revision states`);
        }
      }
    }

    const structuralOccurrences = new Map<string, import('./types').OccurrencePrecondition>();
    const rememberStructuralOccurrence = (
      occurrence: import('./types').OccurrencePrecondition,
    ): void => {
      const existing = structuralOccurrences.get(occurrence.id);
      if (existing && !occurrenceSnapshotEqual(existing, occurrence)) {
        fail(`proposal ${proposalId} has inconsistent structural snapshots for occurrence ${occurrence.id}`);
      }
      structuralOccurrences.set(occurrence.id, occurrence);
    };
    if (p.freshnessStructure !== undefined) {
      if (!isRecord(p.freshnessStructure)) fail(`proposal ${proposalId} freshnessStructure must be an object`);
      const { containers, placements } = p.freshnessStructure;
      if (!Array.isArray(containers) || !Array.isArray(placements)) {
        fail(`proposal ${proposalId} freshnessStructure must contain arrays`);
      }
      const seenContainers = new Set<string>();
      for (const entry of containers) {
        if (!isRecord(entry) || !Array.isArray(entry.occurrences)) {
          fail(`proposal ${proposalId} has an invalid container precondition`);
        }
        const containerId = requireNonEmptyString(
          entry.containerId,
          `proposal ${proposalId} structure containerId`,
        );
        if (seenContainers.has(containerId)) {
          fail(`proposal ${proposalId} repeats structure container ${containerId}`);
        }
        seenContainers.add(containerId);
        if (!state.chunks.has(containerId)) {
          fail(`proposal ${proposalId} structure names unknown container ${containerId}`);
        }
        const seenOccurrences = new Set<string>();
        const expected = entry.occurrences.map((value, index) => {
          const occurrence = validateOccurrenceSnapshot(proposalId, value, `container ${containerId}[${index}]`);
          if (occurrence.containerId !== containerId) {
            fail(`proposal ${proposalId} occurrence ${occurrence.id} is not in container ${containerId}`);
          }
          if (seenOccurrences.has(occurrence.id)) {
            fail(`proposal ${proposalId} repeats occurrence ${occurrence.id} in container ${containerId}`);
          }
          seenOccurrences.add(occurrence.id);
          rememberStructuralOccurrence(occurrence);
          return occurrence;
        });
        for (let index = 1; index < expected.length; index++) {
          if (expected[index - 1].position >= expected[index].position) {
            fail(`proposal ${proposalId} container ${containerId} occurrences are not in position order`);
          }
        }
      }
      const seenPlacements = new Set<string>();
      for (const entry of placements) {
        if (!isRecord(entry) || !Array.isArray(entry.occurrences)) {
          fail(`proposal ${proposalId} has an invalid placement precondition`);
        }
        const chunkId = requireNonEmptyString(entry.chunkId, `proposal ${proposalId} placement chunkId`);
        if (seenPlacements.has(chunkId)) fail(`proposal ${proposalId} repeats placement chunk ${chunkId}`);
        seenPlacements.add(chunkId);
        if (!state.chunks.has(chunkId)) fail(`proposal ${proposalId} placements name unknown chunk ${chunkId}`);
        const seenOccurrences = new Set<string>();
        const expected = entry.occurrences.map((value, index) => {
          const occurrence = validateOccurrenceSnapshot(proposalId, value, `placement ${chunkId}[${index}]`);
          if (occurrence.chunkId !== chunkId) {
            fail(`proposal ${proposalId} occurrence ${occurrence.id} does not place chunk ${chunkId}`);
          }
          if (seenOccurrences.has(occurrence.id)) {
            fail(`proposal ${proposalId} repeats occurrence ${occurrence.id} for chunk ${chunkId}`);
          }
          seenOccurrences.add(occurrence.id);
          rememberStructuralOccurrence(occurrence);
          return occurrence;
        });
        for (let index = 1; index < expected.length; index++) {
          if (expected[index - 1].id >= expected[index].id) {
            fail(`proposal ${proposalId} placements for ${chunkId} are not in occurrence-id order`);
          }
        }
      }
    }

    if (p.producer !== undefined) {
      if (!isRecord(p.producer)) fail(`proposal ${proposalId} producer must be an object`);
      requireNonEmptyString(p.producer.id, `proposal ${proposalId} producer.id`);
      requireNonEmptyString(p.producer.version, `proposal ${proposalId} producer.version`);
      if (p.producer.implementation !== undefined) {
        requireNonEmptyString(p.producer.implementation, `proposal ${proposalId} producer.implementation`);
      }
      if (p.producer.receiptId !== undefined) {
        requireNonEmptyString(p.producer.receiptId, `proposal ${proposalId} producer.receiptId`);
      }
    }

    if (!Array.isArray(p.payload)) fail(`proposal ${proposalId} payload must be an array`);
    const tempIds = new Set<string>();
    for (const raw of p.payload) {
      if (!isRecord(raw)) fail(`proposal ${proposalId} has a non-object proposed change`);
      if (raw.op === 'create') {
        const tempId = requireNonEmptyString(raw.tempId, `proposal ${proposalId} create.tempId`);
        if (tempIds.has(tempId)) fail(`proposal ${proposalId} creates duplicate tempId ${tempId}`);
        tempIds.add(tempId);
      }
    }
    const revisedChunks = new Set<string>();
    const availableTempIds = new Set<string>();
    const repinnedOccurrences = new Set<string>();
    const severedOccurrences = new Set<string>();
    for (const raw of p.payload) {
      const change = raw as ProposedChange;
      switch (change.op) {
        case 'revise': {
          const chunkId = requireNonEmptyString(change.chunkId, `proposal ${proposalId} revise.chunkId`);
          if (!knownChunk(chunkId)) fail(`proposal ${proposalId} revises unknown chunk ${chunkId}`);
          if (revisedChunks.has(chunkId)) fail(`proposal ${proposalId} revises chunk ${chunkId} more than once`);
          revisedChunks.add(chunkId);
          if (typeof change.text !== 'string') fail(`proposal ${proposalId} revise.text must be a string`);
          if (change.mediaType !== undefined) {
            requireNonEmptyString(change.mediaType, `proposal ${proposalId} revise.mediaType`);
          }
          if (change.mergeParentRevisionIds !== undefined) {
            const parents = requireUniqueStrings(
              change.mergeParentRevisionIds,
              `proposal ${proposalId} mergeParentRevisionIds`,
            );
            if (parents.length === 0) fail(`proposal ${proposalId} merge parents cannot be empty`);
            for (const revisionId of parents) {
              const revision = revisionOf(revisionId);
              if (!revision) fail(`proposal ${proposalId} has unknown merge parent ${revisionId}`);
              if (revision.chunkId !== chunkId) {
                fail(`proposal ${proposalId} merge parent ${revisionId} belongs to ${revision.chunkId}, not ${chunkId}`);
              }
              if (!basis.has(revisionId)) {
                fail(`proposal ${proposalId} merge parent ${revisionId} is absent from its basis`);
              }
            }
          } else if (![...basis].some((revisionId) => revisionOf(revisionId)?.chunkId === chunkId)) {
            fail(`proposal ${proposalId} revise basis has no revision of chunk ${chunkId}`);
          }
          break;
        }
        case 'create': {
          availableTempIds.add(change.tempId);
          if (typeof change.text !== 'string') fail(`proposal ${proposalId} create.text must be a string`);
          if (change.mediaType !== undefined) {
            requireNonEmptyString(change.mediaType, `proposal ${proposalId} create.mediaType`);
          }
          if (change.derivedFrom !== undefined) {
            if (!isRecord(change.derivedFrom)) fail(`proposal ${proposalId} create.derivedFrom must be an object`);
            const sourceRevisionId = requireNonEmptyString(
              change.derivedFrom.sourceRevisionId,
              `proposal ${proposalId} derived sourceRevisionId`,
            );
            if (!knownRevision(sourceRevisionId)) {
              fail(`proposal ${proposalId} derives from unknown revision ${sourceRevisionId}`);
            }
            if (!basis.has(sourceRevisionId)) {
              fail(`proposal ${proposalId} derived revision ${sourceRevisionId} is absent from its basis`);
            }
            if (!['copy', 'fork', 'generate', 'extract'].includes(change.derivedFrom.via)) {
              fail(`proposal ${proposalId} has invalid derivation method ${String(change.derivedFrom.via)}`);
            }
            if (change.derivedFrom.sourceSpan !== undefined) {
              validateSpan(
                change.derivedFrom.sourceSpan,
                `proposal ${proposalId} derived sourceSpan`,
                revisionOf,
                blobTextOf,
                sourceRevisionId,
              );
            }
          }
          break;
        }
        case 'place': {
          const containerId = requireNonEmptyString(change.containerId, `proposal ${proposalId} place.containerId`);
          if (!knownChunk(containerId)) fail(`proposal ${proposalId} places into unknown container ${containerId}`);
          let placedChunkId: string | undefined;
          if (typeof change.chunkId === 'string') {
            placedChunkId = requireNonEmptyString(change.chunkId, `proposal ${proposalId} place.chunkId`);
            if (!knownChunk(placedChunkId)) fail(`proposal ${proposalId} places unknown chunk ${placedChunkId}`);
          } else if (isRecord(change.chunkId)) {
            const tempId = requireNonEmptyString(change.chunkId.tempId, `proposal ${proposalId} place.tempId`);
            if (!tempIds.has(tempId)) fail(`proposal ${proposalId} place references unknown tempId ${tempId}`);
            if (!availableTempIds.has(tempId)) {
              fail(`proposal ${proposalId} place references tempId ${tempId} before it is created`);
            }
          } else {
            fail(`proposal ${proposalId} place.chunkId is invalid`);
          }
          if (change.after !== undefined && change.at !== undefined) {
            fail(`proposal ${proposalId} place cannot specify both after and at`);
          }
          if (change.at !== undefined && change.at !== 'start') {
            fail(`proposal ${proposalId} place has invalid at value`);
          }
          if (change.after !== undefined) {
            const after = requireNonEmptyString(change.after, `proposal ${proposalId} place.after`);
            const anchor = state.occurrences.get(after);
            const historicalAnchor = structuralOccurrences.get(after);
            if (!anchor && !historicalAnchor) fail(`proposal ${proposalId} place anchor ${after} is unknown`);
            const anchorContainerId = anchor?.containerId ?? historicalAnchor!.containerId;
            if (anchorContainerId !== containerId) {
              fail(`proposal ${proposalId} place anchor ${after} belongs to ${anchorContainerId}, not ${containerId}`);
            }
          }
          if (change.mode !== undefined && change.mode !== 'contain' && change.mode !== 'transclude') {
            fail(`proposal ${proposalId} place has invalid mode`);
          }
          if (change.watch !== undefined && typeof change.watch !== 'boolean') {
            fail(`proposal ${proposalId} place.watch must be boolean`);
          }
          break;
        }
        case 'repin': {
          const occurrenceId = requireNonEmptyString(
            change.occurrenceId,
            `proposal ${proposalId} repin.occurrenceId`,
          );
          const occurrence = state.occurrences.get(occurrenceId) ?? structuralOccurrences.get(occurrenceId);
          if (!occurrence) fail(`proposal ${proposalId} repins unknown occurrence ${occurrenceId}`);
          if (repinnedOccurrences.has(occurrenceId)) {
            fail(`proposal ${proposalId} repins occurrence ${occurrenceId} more than once`);
          }
          if (severedOccurrences.has(occurrenceId)) {
            fail(`proposal ${proposalId} both repins and severs occurrence ${occurrenceId}`);
          }
          repinnedOccurrences.add(occurrenceId);
          const revisionId = requireNonEmptyString(change.revisionId, `proposal ${proposalId} repin.revisionId`);
          const revision = revisionOf(revisionId);
          if (!revision) fail(`proposal ${proposalId} repins to unknown revision ${revisionId}`);
          if (revision.chunkId !== occurrence.chunkId) {
            fail(`proposal ${proposalId} repin revision ${revisionId} belongs to ${revision.chunkId}, not ${occurrence.chunkId}`);
          }
          if (occurrence.pin !== 'current' && !basis.has(occurrence.pin)) {
            fail(`proposal ${proposalId} repin basis does not include pin ${occurrence.pin}`);
          }
          break;
        }
        case 'relate': {
          const fromChunkId = requireNonEmptyString(
            change.fromChunkId,
            `proposal ${proposalId} relate.fromChunkId`,
          );
          if (!knownChunk(fromChunkId)) fail(`proposal ${proposalId} relates from unknown chunk ${fromChunkId}`);
          const hasChunk = change.toChunkId !== undefined;
          const hasExternal = change.toExternal !== undefined;
          if (hasChunk === hasExternal) {
            fail(`proposal ${proposalId} relation must have exactly one destination`);
          }
          if (change.toChunkId !== undefined) {
            const toChunkId = requireNonEmptyString(
              change.toChunkId,
              `proposal ${proposalId} relate.toChunkId`,
            );
            if (!knownChunk(toChunkId)) fail(`proposal ${proposalId} relates to unknown chunk ${toChunkId}`);
          }
          if (change.toExternal !== undefined) {
            if (!isRecord(change.toExternal)) fail(`proposal ${proposalId} relate.toExternal must be an object`);
            requireNonEmptyString(change.toExternal.layer, `proposal ${proposalId} external layer`);
            requireNonEmptyString(change.toExternal.key, `proposal ${proposalId} external key`);
            if (change.toExternal.url !== undefined) {
              requireNonEmptyString(change.toExternal.url, `proposal ${proposalId} external url`);
            }
            if (change.toExternal.snapshotAt !== undefined) {
              requireNonEmptyString(change.toExternal.snapshotAt, `proposal ${proposalId} external snapshotAt`);
            }
          }
          requireNonEmptyString(change.role, `proposal ${proposalId} relation role`);
          break;
        }
        case 'sever': {
          const occurrenceId = requireNonEmptyString(
            change.occurrenceId,
            `proposal ${proposalId} sever.occurrenceId`,
          );
          if (!state.occurrences.has(occurrenceId) && !structuralOccurrences.has(occurrenceId)) {
            fail(`proposal ${proposalId} severs unknown occurrence ${occurrenceId}`);
          }
          if (severedOccurrences.has(occurrenceId)) {
            fail(`proposal ${proposalId} severs occurrence ${occurrenceId} more than once`);
          }
          if (repinnedOccurrences.has(occurrenceId)) {
            fail(`proposal ${proposalId} both repins and severs occurrence ${occurrenceId}`);
          }
          severedOccurrences.add(occurrenceId);
          break;
        }
        default:
          fail(`proposal ${proposalId} has unknown change op ${String((raw as { op?: unknown }).op)}`);
      }
    }

    if (!isRecord(commit.operation.params) || commit.operation.params.kind === undefined) {
      fail(`proposal ${proposalId} propose operation must name its kind`);
    }
    if (commit.operation.params.kind !== p.kind) {
      fail(`proposal ${proposalId} kind disagrees with its propose operation`);
    }
    if (!sameProducer(p.producer, commit.operation.params.producer)) {
      fail(`proposal ${proposalId} producer disagrees with its propose operation`);
    }
  }

  const updateIds = new Set<string>();
  for (const u of f.proposalUpdates ?? []) {
    const id = requireNonEmptyString(u.id, 'proposalUpdate id');
    if (updateIds.has(id)) fail(`proposalUpdate: duplicate update for proposal ${id}`);
    updateIds.add(id);
    const prior = state.proposals.get(id);
    if (!prior) fail(`proposalUpdate: unknown proposal ${id}`);
    if (prior.status !== 'open') {
      fail(`proposalUpdate: proposal ${id} is already ${prior.status}`);
    }
    if (u.status !== 'accepted' && u.status !== 'rejected' && u.status !== 'superseded') {
      fail(`proposalUpdate: proposal ${id} cannot transition from open to ${String(u.status)}`);
    }
    const expectedOperationKind = u.status === 'accepted' ? 'accept' : 'reject';
    if (commit.operation.kind !== expectedOperationKind) {
      fail(`proposalUpdate: ${u.status} requires ${expectedOperationKind} operation, not ${commit.operation.kind}`);
    }
    if (u.resolution === undefined) {
      fail(`proposalUpdate: terminal transition for ${id} requires a resolution`);
    }
    if (!isRecord(u.resolution)) fail(`proposalUpdate: resolution for ${id} must be an object`);
    if (u.resolution.by !== commit.operation.actorId || u.resolution.at !== commit.operation.at) {
      fail(`proposalUpdate: resolution for ${id} is not bound to the current operation actor and time`);
    }
    requireNonEmptyString(u.resolution.operationId, `proposalUpdate: resolution for ${id}.operationId`);
    if (u.resolution.operationId !== commit.operation.id) {
      fail(`proposalUpdate: resolution for ${id} belongs to operation ${u.resolution.operationId}, not ${commit.operation.id}`);
    }
    if (u.resolution.reason !== undefined && typeof u.resolution.reason !== 'string') {
      fail(`proposalUpdate: resolution reason for ${id} must be a string`);
    }
    if (isRecord(commit.operation.params) && commit.operation.params.proposalId !== undefined &&
      commit.operation.params.proposalId !== id) {
      fail(`proposalUpdate: operation names proposal ${String(commit.operation.params.proposalId)}, not ${id}`);
    }
  }
  if (commit.operation.kind === 'propose') {
    if (proposals.length === 0) fail(`propose operation must carry at least one proposal`);
    requireOnlyFactKinds(commit, new Set(['proposals']));
  } else if (commit.operation.kind === 'reject') {
    if ((f.proposalUpdates ?? []).length !== 1) {
      fail(`reject operation must resolve exactly one proposal`);
    }
    requireOnlyFactKinds(commit, new Set(['proposalUpdates']));
  } else if (commit.operation.kind === 'accept') {
    const updates = f.proposalUpdates ?? [];
    if (updates.length !== 1 || updates[0].status !== 'accepted') {
      fail(`accept operation must accept exactly one proposal`);
    }
    const proposal = state.proposals.get(updates[0].id)!;
    validateAcceptedRealization(state, commit, proposal);
  }
  for (const id of f.tombstone ?? []) {
    if (!state.chunks.has(id) && !newChunks.has(id)) fail(`tombstone: unknown chunk ${id}`);
  }
  for (const id of f.redactRevisions ?? []) {
    if (!state.revisions.has(id) && !newRevisions.has(id)) fail(`redact: unknown revision ${id}`);
  }
}

// Fold a validated commit into state. Total after validateCommit: every lookup
// here was proven present above, so this cannot fail partway and tear state.
// Mutates in place; callers that need react-style change detection bump their
// own version counter per commit.
export function foldCommit(state: WorkspaceGraph, commit: Commit): void {
  const f = commit.facts;
  for (const b of f.blobs ?? []) state.blobs.set(b.hash, b);
  for (const c of f.chunks ?? []) state.chunks.set(c.id, { ...c });
  for (const r of f.revisions ?? []) state.revisions.set(r.id, r);
  for (const s of f.setCurrent ?? []) state.chunks.get(s.chunkId)!.currentRevisionId = s.revisionId;
  for (const o of f.occurrences ?? []) state.occurrences.set(o.id, { ...o });
  for (const u of f.occurrenceUpdates ?? []) {
    const occ = state.occurrences.get(u.id)!;
    if (u.position !== undefined) occ.position = u.position;
    if (u.pin !== undefined) occ.pin = u.pin;
    if (u.watch !== undefined) occ.watch = u.watch;
  }
  for (const id of f.removeOccurrences ?? []) state.occurrences.delete(id);
  for (const l of f.links ?? []) state.links.set(l.id, l);
  for (const id of f.removeLinks ?? []) state.links.delete(id);
  for (const d of f.derivations ?? []) state.derivations.set(d.id, d);
  for (const p of f.proposals ?? []) state.proposals.set(p.id, { ...p });
  for (const u of f.proposalUpdates ?? []) {
    const p = state.proposals.get(u.id)!;
    p.status = u.status;
    if (u.resolution) p.resolution = u.resolution;
  }
  for (const id of f.tombstone ?? []) state.chunks.get(id)!.tombstoned = true;
  for (const id of f.redactRevisions ?? []) state.revisions.set(id, { ...state.revisions.get(id)!, redacted: true });
  state.operations.set(commit.operation.id, commit.operation);
  state.head = commit.id;
  state.commitCount++;
}

// Validate and fold in one step, for callers with nothing to persist between.
export function applyCommit(
  state: WorkspaceGraph,
  commit: Commit,
): void {
  validateCommit(state, commit);
  foldCommit(state, commit);
}

export function materialize(commits: Iterable<Commit>): WorkspaceGraph {
  const state = emptyState();
  for (const c of commits) applyCommit(state, c);
  return state;
}

// ── accessors ────────────────────────────────────────────────────────────────

export function currentRevision(state: WorkspaceGraph, chunkId: ChunkId): Revision {
  const chunk = state.chunks.get(chunkId) ?? fail(`unknown chunk ${chunkId}`);
  return state.revisions.get((chunk as Chunk).currentRevisionId) ?? fail(`chunk ${chunkId} head revision missing`);
}

export function revisionText(state: WorkspaceGraph, revisionId: RevisionId): string {
  const rev = state.revisions.get(revisionId) ?? fail(`unknown revision ${revisionId}`);
  if ((rev as Revision).redacted) return '[redacted]';
  const blob = state.blobs.get((rev as Revision).blobHash) ?? fail(`revision ${revisionId} blob missing`);
  return blob.text;
}

export function childOccurrences(state: WorkspaceGraph, containerId: ChunkId): Occurrence[] {
  const out: Occurrence[] = [];
  for (const occ of state.occurrences.values()) if (occ.containerId === containerId) out.push(occ);
  return out.sort((a, b) => (a.position < b.position ? -1 : a.position > b.position ? 1 : 0));
}

export function occurrencesOfChunk(state: WorkspaceGraph, chunkId: ChunkId): Occurrence[] {
  const out: Occurrence[] = [];
  for (const occ of state.occurrences.values()) if (occ.chunkId === chunkId) out.push(occ);
  return out;
}

// Why an open proposal can no longer be admitted. Kept in the state layer so
// both the transaction builder and hostile-commit validator use exactly the
// same applicability test.
export function proposalStaleReason(state: WorkspaceGraph, p: Proposal): string | null {
  if (p.freshnessRevisionStates) {
    for (const expected of p.freshnessRevisionStates) {
      const revision = state.revisions.get(expected.revisionId);
      if (!revision || revision.chunkId !== expected.chunkId) {
        return `context revision ${expected.revisionId} no longer exists`;
      }
      const chunk = state.chunks.get(expected.chunkId);
      if (!chunk || chunk.tombstoned !== expected.chunkTombstoned) {
        return `context chunk ${expected.chunkId} visibility has changed since selection`;
      }
      if (Boolean(revision.redacted) !== expected.redacted) {
        return `context revision ${expected.revisionId} visibility has changed since selection`;
      }
      if (expected.followsCurrent && chunk.currentRevisionId !== expected.revisionId) {
        return `context chunk ${expected.chunkId} has moved on since it was selected`;
      }
    }
  } else {
    for (const revisionId of p.freshnessRevisionIds ?? []) {
      const revision = state.revisions.get(revisionId);
      if (!revision) return `context revision ${revisionId} no longer exists`;
      const chunk = state.chunks.get(revision.chunkId);
      if (!chunk || chunk.tombstoned) return `context chunk ${revision.chunkId} no longer exists`;
      if (chunk.currentRevisionId !== revisionId) {
        return `context chunk ${revision.chunkId} has moved on since it was selected`;
      }
    }
  }
  const sameOccurrences = (
    actual: Occurrence[],
    expected: NonNullable<Proposal['freshnessStructure']>['containers'][number]['occurrences'],
  ): boolean =>
    actual.length === expected.length && actual.every((occurrence, index) => {
      const prior = expected[index];
      return prior !== undefined &&
        occurrence.id === prior.id &&
        occurrence.containerId === prior.containerId &&
        occurrence.chunkId === prior.chunkId &&
        occurrence.position === prior.position &&
        occurrence.mode === prior.mode &&
        occurrence.pin === prior.pin &&
        occurrence.watch === prior.watch;
    });
  for (const precondition of p.freshnessStructure?.containers ?? []) {
    if (!state.chunks.has(precondition.containerId)) {
      return `context container ${precondition.containerId} no longer exists`;
    }
    if (!sameOccurrences(childOccurrences(state, precondition.containerId), precondition.occurrences)) {
      return `context structure in ${precondition.containerId} has changed since selection`;
    }
  }
  for (const precondition of p.freshnessStructure?.placements ?? []) {
    if (!state.chunks.has(precondition.chunkId)) {
      return `context chunk ${precondition.chunkId} no longer exists`;
    }
    const actual = occurrencesOfChunk(state, precondition.chunkId).sort((a, b) => a.id.localeCompare(b.id));
    if (!sameOccurrences(actual, precondition.occurrences)) {
      return `context placement for ${precondition.chunkId} has changed since selection`;
    }
  }
  for (const change of p.payload) {
    if (change.op === 'revise') {
      const chunk = state.chunks.get(change.chunkId);
      if (!chunk) return `target chunk ${change.chunkId} no longer exists`;
      if (change.mergeParentRevisionIds) {
        if (!change.mergeParentRevisionIds.includes(chunk.currentRevisionId)) {
          return `chunk ${change.chunkId} advanced past the merge parents`;
        }
      } else if (!p.basisRevisionIds.includes(chunk.currentRevisionId)) {
        return `chunk ${change.chunkId} has moved on since the proposal's basis`;
      }
    } else if (change.op === 'repin') {
      const occurrence = state.occurrences.get(change.occurrenceId);
      if (!occurrence) return `occurrence ${change.occurrenceId} no longer exists`;
      const revision = state.revisions.get(change.revisionId);
      if (!revision || revision.chunkId !== occurrence.chunkId) {
        return `repin target ${change.revisionId} is no longer valid for occurrence ${change.occurrenceId}`;
      }
      if (occurrence.pin === 'current') return `occurrence ${change.occurrenceId} follows current; nothing to update`;
      if (!p.basisRevisionIds.includes(occurrence.pin)) {
        return `occurrence ${change.occurrenceId} was repinned since the proposal's basis`;
      }
    } else if (change.op === 'sever') {
      if (!state.occurrences.has(change.occurrenceId)) return `occurrence ${change.occurrenceId} already severed`;
    } else if (change.op === 'place') {
      if (typeof change.chunkId === 'string' && !state.chunks.has(change.chunkId)) {
        return `chunk ${change.chunkId} no longer exists`;
      }
      if (!state.chunks.has(change.containerId)) return `container ${change.containerId} no longer exists`;
      if (change.after !== undefined) {
        const anchor = state.occurrences.get(change.after);
        if (!anchor) return `anchor occurrence ${change.after} no longer exists`;
        if (anchor.containerId !== change.containerId) {
          return `anchor occurrence ${change.after} is no longer in container ${change.containerId}`;
        }
      }
      if (typeof change.chunkId === 'string' && wouldCreateCycle(state, change.containerId, change.chunkId)) {
        return `placing chunk ${change.chunkId} in ${change.containerId} would now create a containment cycle`;
      }
    }
  }
  if (
    p.payload.length > 0 &&
    p.targetChunkIds.length > 0 &&
    p.basisRevisionIds.length > 0 &&
    p.payload.every((change) => change.op === 'create' || change.op === 'place' || change.op === 'relate')
  ) {
    const anchored = p.targetChunkIds.some((target) => {
      const chunk = state.chunks.get(target);
      return chunk !== undefined && p.basisRevisionIds.includes(chunk.currentRevisionId);
    });
    if (!anchored) return `no target chunk is still at the proposal's basis`;
  }
  return null;
}

// The revision an occurrence renders: its pin, or the chunk's current head.
export function occurrenceRevision(state: WorkspaceGraph, occ: Occurrence): Revision {
  if (occ.pin === 'current') return currentRevision(state, occ.chunkId);
  return (state.revisions.get(occ.pin) as Revision) ?? fail(`occurrence ${occ.id} pinned to missing revision ${occ.pin}`);
}

export function isComposite(state: WorkspaceGraph, chunkId: ChunkId): boolean {
  return isCompositeMediaType(currentRevision(state, chunkId).mediaType);
}

// A composite blob's text is JSON: { "join": string }. The separator is part of
// how the content reads (content), while ordering stays in occurrences
// (arrangement) — so rearranging never edits content, but rendering is exact.
export function compositeJoin(state: WorkspaceGraph, rev: Revision): string {
  try {
    const parsed = JSON.parse(state.blobs.get(rev.blobHash)?.text ?? '');
    if (typeof parsed?.join === 'string') return parsed.join;
  } catch {
    /* fall through to default */
  }
  return '\n\n';
}

// Reader's text of an exact revision. Composite arrangement is still the
// chunk's occurrence set, while its pinned revision determines visibility,
// media type, and join semantics. Nested occurrences likewise render their
// own effective (current or pinned) revision.
export function renderRevision(state: WorkspaceGraph, revisionId: RevisionId, seen: Set<ChunkId> = new Set()): string {
  const rev = state.revisions.get(revisionId) ?? fail(`unknown revision ${revisionId}`);
  const chunkId = (rev as Revision).chunkId;
  if (seen.has(chunkId)) return `[cycle: ${chunkId}]`;
  seen.add(chunkId);
  if ((rev as Revision).redacted) {
    seen.delete(chunkId);
    return '[redacted]';
  }
  if (!isCompositeMediaType(rev.mediaType)) {
    seen.delete(chunkId);
    return revisionText(state, rev.id);
  }
  const parts = childOccurrences(state, chunkId).map((occ) => {
    const r = occurrenceRevision(state, occ);
    return renderRevision(state, r.id, seen);
  });
  seen.delete(chunkId);
  return parts.join(compositeJoin(state, rev));
}

// Reader's text of a continuing chunk identity at its current revision.
export function renderChunk(state: WorkspaceGraph, chunkId: ChunkId, seen: Set<ChunkId> = new Set()): string {
  return renderRevision(state, currentRevision(state, chunkId).id, seen);
}

export function openProposals(state: WorkspaceGraph, targetChunkId?: ChunkId): Proposal[] {
  const out: Proposal[] = [];
  for (const p of state.proposals.values()) {
    if (p.status !== 'open') continue;
    if (targetChunkId && !p.targetChunkIds.includes(targetChunkId)) continue;
    out.push(p);
  }
  return out.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}
