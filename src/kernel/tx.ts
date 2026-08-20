// Transactions: every mutation is one operation, one commit, one log append.
// Builders validate against current state, construct explicit facts, prove the
// commit legal, hand it to the persistence hook, and only then advance memory.

import { keyBetween, keysBetween } from './fractional';
import { makeBlob } from './hash';
import {
  newChunkId,
  newCommitId,
  newDerivationId,
  newLinkId,
  newOccurrenceId,
  newOperationId,
  newProposalId,
  newRevisionId,
} from './ids';
import {
  childOccurrences,
  currentRevision,
  foldCommit,
  occurrencesOfChunk,
  revisionText,
  validateCommit,
  type SubstrateState,
} from './state';
import type {
  ActorId,
  ChunkId,
  Commit,
  Derivation,
  ExternalRef,
  Facts,
  Link,
  Occurrence,
  OccurrenceId,
  Operation,
  OperationKind,
  Proposal,
  ProposalId,
  ProposalKind,
  ProposedChange,
  Revision,
  RevisionId,
  SpanAddress,
} from './types';
import { MEDIA_COMPOSITE, MEDIA_TEXT } from './types';

export interface TxCtx {
  state: SubstrateState;
  actorId: ActorId;
  now?: () => string;
  // Durability hook: the commit is validated but NOT yet folded. Throwing here
  // aborts the transaction with state untouched, which is the whole point of
  // the ordering — persist first, advance memory second.
  onCommit?: (commit: Commit) => void;
  // The commit is truth now. For work that depends on the folded state and
  // must not fail it: snapshot cadence, view invalidation, dispatch.
  afterCommit?: (commit: Commit) => void;
  // Permission seam: checked before any commit. Absent policy = local allow-all.
  policy?: (kind: OperationKind, targetChunkIds: ChunkId[]) => boolean;
}

export class PermissionError extends Error {}

const t = (ctx: TxCtx) => (ctx.now ?? (() => new Date().toISOString()))();

function finish(
  ctx: TxCtx,
  op: { id: string; kind: OperationKind; at: string; inputs?: RevisionId[]; outputs?: RevisionId[]; params?: unknown },
  facts: Facts,
  targets: ChunkId[] = [],
): Commit {
  if (ctx.policy && !ctx.policy(op.kind, targets)) {
    throw new PermissionError(`actor ${ctx.actorId} may not ${op.kind} [${targets.join(', ')}]`);
  }
  const operation: Operation = {
    id: op.id,
    kind: op.kind,
    actorId: ctx.actorId,
    at: op.at,
    inputRevisionIds: op.inputs ?? [],
    outputRevisionIds: op.outputs ?? [],
    params: op.params,
  };
  const commit: Commit = {
    id: newCommitId(),
    parentIds: ctx.state.head ? [ctx.state.head] : [],
    at: op.at,
    actorId: ctx.actorId,
    operation,
    facts,
  };
  // Durability ordering. Validation proves the commit legal without touching
  // state; the hook makes it durable; the fold cannot fail after that. So a
  // persistence error leaves memory exactly where the log is, never ahead of
  // it — the divergence that would otherwise poison the next snapshot.
  validateCommit(ctx.state, commit);
  ctx.onCommit?.(commit);
  foldCommit(ctx.state, commit);
  ctx.afterCommit?.(commit);
  return commit;
}

// ── positions ────────────────────────────────────────────────────────────────

export type PlaceAt = 'start' | 'end' | { after: OccurrenceId };

function positionIn(state: SubstrateState, containerId: ChunkId, at: PlaceAt = 'end'): string {
  const sibs = childOccurrences(state, containerId);
  if (at === 'end') return keyBetween(sibs.length ? sibs[sibs.length - 1].position : null, null);
  if (at === 'start') return keyBetween(null, sibs.length ? sibs[0].position : null);
  const i = sibs.findIndex((o) => o.id === at.after);
  if (i < 0) throw new Error(`positionIn: occurrence ${at.after} not in container ${containerId}`);
  return keyBetween(sibs[i].position, sibs[i + 1]?.position ?? null);
}

// ── create / revise ──────────────────────────────────────────────────────────

export async function createChunk(
  ctx: TxCtx,
  opts: { text: string; mediaType?: string; containerId?: ChunkId; at?: PlaceAt },
): Promise<{ chunkId: ChunkId; revisionId: RevisionId; occurrenceId?: OccurrenceId; commit: Commit }> {
  const opId = newOperationId();
  const when = t(ctx);
  const mediaType = opts.mediaType ?? MEDIA_TEXT;
  const blob = await makeBlob(mediaType, opts.text);
  const chunkId = newChunkId();
  const revision: Revision = {
    id: newRevisionId(),
    chunkId,
    blobHash: blob.hash,
    mediaType,
    parentRevisionIds: [],
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  const facts: Facts = {
    blobs: [blob],
    chunks: [{ id: chunkId, currentRevisionId: revision.id, tombstoned: false }],
    revisions: [revision],
  };
  let occurrenceId: OccurrenceId | undefined;
  if (opts.containerId) {
    occurrenceId = newOccurrenceId();
    facts.occurrences = [
      {
        id: occurrenceId,
        containerId: opts.containerId,
        chunkId,
        position: positionIn(ctx.state, opts.containerId, opts.at),
        mode: 'contain',
        pin: 'current',
        watch: false,
      },
    ];
  }
  const commit = finish(ctx, { id: opId, kind: 'create', at: when, outputs: [revision.id] }, facts, [
    ...(opts.containerId ? [opts.containerId] : []),
  ]);
  return { chunkId, revisionId: revision.id, occurrenceId, commit };
}

export async function revise(
  ctx: TxCtx,
  opts: { chunkId: ChunkId; text: string; mediaType?: string; mergeParentRevisionIds?: RevisionId[] },
): Promise<{ revisionId: RevisionId; commit: Commit }> {
  const opId = newOperationId();
  const when = t(ctx);
  const cur = currentRevision(ctx.state, opts.chunkId);
  const mediaType = opts.mediaType ?? cur.mediaType;
  const blob = await makeBlob(mediaType, opts.text);
  const parents = opts.mergeParentRevisionIds ?? [cur.id];
  const revision: Revision = {
    id: newRevisionId(),
    chunkId: opts.chunkId,
    blobHash: blob.hash,
    mediaType,
    parentRevisionIds: parents,
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  const commit = finish(
    ctx,
    { id: opId, kind: 'revise', at: when, inputs: parents, outputs: [revision.id] },
    { blobs: [blob], revisions: [revision], setCurrent: [{ chunkId: opts.chunkId, revisionId: revision.id }] },
    [opts.chunkId],
  );
  return { revisionId: revision.id, commit };
}

// One atomic commit for a whole composite: the container, every child, and
// their placements. This is what drivers use so an import can never be seen
// (or persisted) half-done.
export async function createComposite(
  ctx: TxCtx,
  opts: {
    join: string;
    blocks: { text: string; mediaType?: string }[];
    containerId?: ChunkId;
    at?: PlaceAt;
    opKind?: 'create' | 'import';
  },
): Promise<{ chunkId: ChunkId; revisionId: RevisionId; blockChunkIds: ChunkId[]; commit: Commit }> {
  const opId = newOperationId();
  const when = t(ctx);
  const facts: Facts = { blobs: [], chunks: [], revisions: [], occurrences: [] };

  const compositeBlob = await makeBlob(MEDIA_COMPOSITE, JSON.stringify({ join: opts.join }));
  const chunkId = newChunkId();
  const revision: Revision = {
    id: newRevisionId(),
    chunkId,
    blobHash: compositeBlob.hash,
    mediaType: MEDIA_COMPOSITE,
    parentRevisionIds: [],
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  facts.blobs!.push(compositeBlob);
  facts.chunks!.push({ id: chunkId, currentRevisionId: revision.id, tombstoned: false });
  facts.revisions!.push(revision);

  const blockChunkIds: ChunkId[] = [];
  const positions = keysBetween(null, null, opts.blocks.length);
  for (const [i, block] of opts.blocks.entries()) {
    const mediaType = block.mediaType ?? MEDIA_TEXT;
    const blob = await makeBlob(mediaType, block.text);
    const blockChunkId = newChunkId();
    facts.blobs!.push(blob);
    facts.chunks!.push({ id: blockChunkId, currentRevisionId: '', tombstoned: false });
    const blockRev: Revision = {
      id: newRevisionId(),
      chunkId: blockChunkId,
      blobHash: blob.hash,
      mediaType,
      parentRevisionIds: [],
      createdBy: ctx.actorId,
      createdAt: when,
      operationId: opId,
    };
    facts.chunks![facts.chunks!.length - 1].currentRevisionId = blockRev.id;
    facts.revisions!.push(blockRev);
    facts.occurrences!.push({
      id: newOccurrenceId(),
      containerId: chunkId,
      chunkId: blockChunkId,
      position: positions[i],
      mode: 'contain',
      pin: 'current',
      watch: false,
    });
    blockChunkIds.push(blockChunkId);
  }
  if (opts.containerId) {
    facts.occurrences!.push({
      id: newOccurrenceId(),
      containerId: opts.containerId,
      chunkId,
      position: positionIn(ctx.state, opts.containerId, opts.at),
      mode: 'contain',
      pin: 'current',
      watch: false,
    });
  }
  const commit = finish(
    ctx,
    { id: opId, kind: opts.opKind ?? 'import', at: when, outputs: facts.revisions!.map((r) => r.id), params: { blocks: opts.blocks.length } },
    facts,
    opts.containerId ? [opts.containerId] : [],
  );
  return { chunkId, revisionId: revision.id, blockChunkIds, commit };
}

// ── arrangement ──────────────────────────────────────────────────────────────

export function placeOccurrence(
  ctx: TxCtx,
  opts: { containerId: ChunkId; chunkId: ChunkId; at?: PlaceAt; mode?: 'contain' | 'transclude'; pin?: 'current' | RevisionId; watch?: boolean },
): { occurrenceId: OccurrenceId; commit: Commit } {
  const opId = newOperationId();
  const when = t(ctx);
  const occ: Occurrence = {
    id: newOccurrenceId(),
    containerId: opts.containerId,
    chunkId: opts.chunkId,
    position: positionIn(ctx.state, opts.containerId, opts.at),
    mode: opts.mode ?? 'contain',
    pin: opts.pin ?? 'current',
    watch: opts.watch ?? false,
  };
  const commit = finish(ctx, { id: opId, kind: 'place', at: when }, { occurrences: [occ] }, [opts.containerId]);
  return { occurrenceId: occ.id, commit };
}

export function moveOccurrence(ctx: TxCtx, opts: { occurrenceId: OccurrenceId; at: PlaceAt }): Commit {
  const occ = ctx.state.occurrences.get(opts.occurrenceId);
  if (!occ) throw new Error(`moveOccurrence: unknown occurrence ${opts.occurrenceId}`);
  // Position is computed among current siblings minus the moving occurrence.
  const sibs = childOccurrences(ctx.state, occ.containerId).filter((o) => o.id !== occ.id);
  let position: string;
  if (opts.at === 'end') position = keyBetween(sibs.length ? sibs[sibs.length - 1].position : null, null);
  else if (opts.at === 'start') position = keyBetween(null, sibs.length ? sibs[0].position : null);
  else {
    const i = sibs.findIndex((o) => o.id === (opts.at as { after: OccurrenceId }).after);
    if (i < 0) throw new Error(`moveOccurrence: anchor not among siblings`);
    position = keyBetween(sibs[i].position, sibs[i + 1]?.position ?? null);
  }
  return finish(
    ctx,
    { id: newOperationId(), kind: 'move', at: t(ctx) },
    { occurrenceUpdates: [{ id: occ.id, position }] },
    [occ.containerId],
  );
}

export function severOccurrence(ctx: TxCtx, opts: { occurrenceId: OccurrenceId }): Commit {
  const occ = ctx.state.occurrences.get(opts.occurrenceId);
  if (!occ) throw new Error(`severOccurrence: unknown occurrence ${opts.occurrenceId}`);
  return finish(
    ctx,
    { id: newOperationId(), kind: 'sever', at: t(ctx) },
    { removeOccurrences: [occ.id] },
    [occ.containerId],
  );
}

// ── reuse: copy, reference, transclude ───────────────────────────────────────

export function copyChunk(
  ctx: TxCtx,
  opts: { sourceChunkId: ChunkId; containerId?: ChunkId; at?: PlaceAt },
): { chunkId: ChunkId; revisionId: RevisionId; commit: Commit } {
  const opId = newOperationId();
  const when = t(ctx);
  const src = currentRevision(ctx.state, opts.sourceChunkId);
  const chunkId = newChunkId();
  const revision: Revision = {
    id: newRevisionId(),
    chunkId,
    blobHash: src.blobHash, // same immutable payload; interning, not shared identity
    mediaType: src.mediaType,
    parentRevisionIds: [],
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  const derivation: Derivation = {
    id: newDerivationId(),
    childChunkId: chunkId,
    sourceRevisionId: src.id,
    via: 'copy',
    operationId: opId,
  };
  const facts: Facts = {
    chunks: [{ id: chunkId, currentRevisionId: revision.id, tombstoned: false }],
    revisions: [revision],
    derivations: [derivation],
  };
  if (opts.containerId) {
    facts.occurrences = [
      {
        id: newOccurrenceId(),
        containerId: opts.containerId,
        chunkId,
        position: positionIn(ctx.state, opts.containerId, opts.at),
        mode: 'contain',
        pin: 'current',
        watch: false,
      },
    ];
  }
  const commit = finish(ctx, { id: opId, kind: 'copy', at: when, inputs: [src.id], outputs: [revision.id] }, facts, [
    opts.sourceChunkId,
  ]);
  return { chunkId, revisionId: revision.id, commit };
}

export function reference(
  ctx: TxCtx,
  opts: { fromChunkId: ChunkId; toChunkId?: ChunkId; toRevisionId?: RevisionId; toExternal?: ExternalRef; role?: string; fromSpan?: SpanAddress },
): { linkId: string; commit: Commit } {
  const opId = newOperationId();
  const link: Link = {
    id: newLinkId(),
    fromChunkId: opts.fromChunkId,
    fromSpan: opts.fromSpan,
    toChunkId: opts.toChunkId,
    toRevisionId: opts.toRevisionId,
    toExternal: opts.toExternal,
    role: opts.role ?? 'references',
    operationId: opId,
  };
  const commit = finish(ctx, { id: opId, kind: 'reference', at: t(ctx) }, { links: [link] }, [opts.fromChunkId]);
  return { linkId: link.id, commit };
}

// Watched transclusion is the default: render a pinned revision, watch the source.
export function transclude(
  ctx: TxCtx,
  opts: { containerId: ChunkId; sourceChunkId: ChunkId; at?: PlaceAt; watch?: boolean },
): { occurrenceId: OccurrenceId; commit: Commit } {
  const opId = newOperationId();
  const pinned = currentRevision(ctx.state, opts.sourceChunkId);
  const occ: Occurrence = {
    id: newOccurrenceId(),
    containerId: opts.containerId,
    chunkId: opts.sourceChunkId,
    position: positionIn(ctx.state, opts.containerId, opts.at),
    mode: 'transclude',
    pin: pinned.id,
    watch: opts.watch ?? true,
  };
  const commit = finish(
    ctx,
    { id: opId, kind: 'transclude', at: t(ctx), inputs: [pinned.id] },
    { occurrences: [occ] },
    [opts.containerId, opts.sourceChunkId],
  );
  return { occurrenceId: occ.id, commit };
}

// ── promotion ────────────────────────────────────────────────────────────────

const spanStale = (state: SubstrateState, span: SpanAddress): Revision => {
  const rev = state.revisions.get(span.revisionId);
  if (!rev) throw new Error(`promote: unknown revision ${span.revisionId}`);
  const chunk = state.chunks.get(rev.chunkId)!;
  if (chunk.currentRevisionId !== rev.id) {
    throw new Error(`promote: span addresses revision ${rev.id}, but ${rev.chunkId} has moved on`);
  }
  return rev;
};

// Extract: the span becomes a child chunk; the unextracted remainder becomes
// the minimal sibling chunks; the parent's new revision is a composite that
// joins children with '' so the rendered text is unchanged.
export async function promoteExtract(
  ctx: TxCtx,
  opts: { span: SpanAddress },
): Promise<{ extractedChunkId: ChunkId; partChunkIds: ChunkId[]; commit: Commit }> {
  const opId = newOperationId();
  const when = t(ctx);
  const parentRev = spanStale(ctx.state, opts.span);
  if (parentRev.mediaType === MEDIA_COMPOSITE) throw new Error('promote: cannot extract from a composite; extract from its child');
  const text = revisionText(ctx.state, parentRev.id);
  let { start, end } = opts.span;
  if (!(start >= 0 && end <= text.length && start < end)) throw new Error(`promote: span [${start},${end}) out of range`);
  // Never split a surrogate pair: snap outward so every segment is valid text.
  const isLowSurrogate = (i: number) => {
    const c = text.charCodeAt(i);
    return c >= 0xdc00 && c <= 0xdfff;
  };
  if (start > 0 && isLowSurrogate(start)) start--;
  if (end < text.length && isLowSurrogate(end)) end++;
  const segments = [
    { start: 0, end: start },
    { start, end },
    { start: end, end: text.length },
  ].filter((s) => s.end > s.start);

  const facts: Facts = { blobs: [], chunks: [], revisions: [], occurrences: [], derivations: [], setCurrent: [] };
  const partChunkIds: ChunkId[] = [];
  let extractedChunkId: ChunkId = '';
  // Anything already placed under this chunk keeps its keys; the extracted
  // parts land before it, so keys never collide and the parts render first.
  const existingChildren = childOccurrences(ctx.state, parentRev.chunkId);
  const positions = keysBetween(null, existingChildren.length ? existingChildren[0].position : null, segments.length);
  for (const [i, seg] of segments.entries()) {
    const blob = await makeBlob(parentRev.mediaType, text.slice(seg.start, seg.end));
    const chunkId = newChunkId();
    const revision: Revision = {
      id: newRevisionId(),
      chunkId,
      blobHash: blob.hash,
      mediaType: parentRev.mediaType,
      parentRevisionIds: [],
      createdBy: ctx.actorId,
      createdAt: when,
      operationId: opId,
    };
    facts.blobs!.push(blob);
    facts.chunks!.push({ id: chunkId, currentRevisionId: revision.id, tombstoned: false });
    facts.revisions!.push(revision);
    facts.derivations!.push({
      id: newDerivationId(),
      childChunkId: chunkId,
      sourceRevisionId: parentRev.id,
      sourceSpan: { revisionId: parentRev.id, method: opts.span.method, start: seg.start, end: seg.end },
      via: 'extract',
      operationId: opId,
    });
    facts.occurrences!.push({
      id: newOccurrenceId(),
      containerId: parentRev.chunkId,
      chunkId,
      position: positions[i],
      mode: 'contain',
      pin: 'current',
      watch: false,
    });
    partChunkIds.push(chunkId);
    if (seg.start === start) extractedChunkId = chunkId;
  }
  const compositeBlob = await makeBlob(MEDIA_COMPOSITE, JSON.stringify({ join: '' }));
  const compositeRev: Revision = {
    id: newRevisionId(),
    chunkId: parentRev.chunkId,
    blobHash: compositeBlob.hash,
    mediaType: MEDIA_COMPOSITE,
    parentRevisionIds: [parentRev.id],
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  facts.blobs!.push(compositeBlob);
  facts.revisions!.push(compositeRev);
  facts.setCurrent!.push({ chunkId: parentRev.chunkId, revisionId: compositeRev.id });

  const commit = finish(
    ctx,
    {
      id: opId,
      kind: 'promote',
      at: when,
      inputs: [parentRev.id],
      outputs: [...facts.revisions!.map((r) => r.id)],
      params: { shape: 'extract', span: opts.span },
    },
    facts,
    [parentRev.chunkId],
  );
  return { extractedChunkId, partChunkIds, commit };
}

// Copy-shape promotion: the span becomes a new independent chunk; parent untouched.
export async function promoteCopy(
  ctx: TxCtx,
  opts: { span: SpanAddress; containerId?: ChunkId; at?: PlaceAt },
): Promise<{ chunkId: ChunkId; revisionId: RevisionId; commit: Commit }> {
  const opId = newOperationId();
  const when = t(ctx);
  const srcRev = ctx.state.revisions.get(opts.span.revisionId);
  if (!srcRev) throw new Error(`promote: unknown revision ${opts.span.revisionId}`);
  const text = revisionText(ctx.state, srcRev.id).slice(opts.span.start, opts.span.end);
  const blob = await makeBlob(srcRev.mediaType, text);
  const chunkId = newChunkId();
  const revision: Revision = {
    id: newRevisionId(),
    chunkId,
    blobHash: blob.hash,
    mediaType: srcRev.mediaType,
    parentRevisionIds: [],
    createdBy: ctx.actorId,
    createdAt: when,
    operationId: opId,
  };
  const facts: Facts = {
    blobs: [blob],
    chunks: [{ id: chunkId, currentRevisionId: revision.id, tombstoned: false }],
    revisions: [revision],
    derivations: [
      {
        id: newDerivationId(),
        childChunkId: chunkId,
        sourceRevisionId: srcRev.id,
        sourceSpan: opts.span,
        via: 'copy',
        operationId: opId,
      },
    ],
  };
  if (opts.containerId) {
    facts.occurrences = [
      {
        id: newOccurrenceId(),
        containerId: opts.containerId,
        chunkId,
        position: positionIn(ctx.state, opts.containerId, opts.at),
        mode: 'contain',
        pin: 'current',
        watch: false,
      },
    ];
  }
  const commit = finish(
    ctx,
    { id: opId, kind: 'promote', at: when, inputs: [srcRev.id], outputs: [revision.id], params: { shape: 'copy', span: opts.span } },
    facts,
    [srcRev.chunkId],
  );
  return { chunkId, revisionId: revision.id, commit };
}

// Span-shape promotion: durable addressability without extraction — a link
// anchored at the span, optionally pointing at a target.
export function promoteSpanAnchor(
  ctx: TxCtx,
  opts: { span: SpanAddress; role?: string; toChunkId?: ChunkId; toExternal?: ExternalRef },
): { linkId: string; commit: Commit } {
  const opId = newOperationId();
  const srcRev = ctx.state.revisions.get(opts.span.revisionId);
  if (!srcRev) throw new Error(`promote: unknown revision ${opts.span.revisionId}`);
  const link: Link = {
    id: newLinkId(),
    fromChunkId: srcRev.chunkId,
    fromSpan: opts.span,
    toChunkId: opts.toChunkId,
    toExternal: opts.toExternal,
    role: opts.role ?? 'span-anchor',
    operationId: opId,
  };
  const commit = finish(
    ctx,
    { id: opId, kind: 'promote', at: t(ctx), inputs: [srcRev.id], params: { shape: 'span', span: opts.span } },
    { links: [link] },
    [srcRev.chunkId],
  );
  return { linkId: link.id, commit };
}

// ── governance ───────────────────────────────────────────────────────────────

export function tombstoneChunk(ctx: TxCtx, opts: { chunkId: ChunkId }): Commit {
  if (!ctx.state.chunks.has(opts.chunkId)) throw new Error(`tombstone: unknown chunk ${opts.chunkId}`);
  return finish(ctx, { id: newOperationId(), kind: 'tombstone', at: t(ctx) }, { tombstone: [opts.chunkId] }, [opts.chunkId]);
}

export function redactRevision(ctx: TxCtx, opts: { revisionId: RevisionId }): Commit {
  const rev = ctx.state.revisions.get(opts.revisionId);
  if (!rev) throw new Error(`redact: unknown revision ${opts.revisionId}`);
  return finish(ctx, { id: newOperationId(), kind: 'redact', at: t(ctx) }, { redactRevisions: [rev.id] }, [rev.chunkId]);
}

// ── proposals ────────────────────────────────────────────────────────────────

export function propose(
  ctx: TxCtx,
  opts: {
    kind: ProposalKind;
    basisRevisionIds: RevisionId[];
    targetChunkIds: ChunkId[];
    payload: ProposedChange[];
    note?: string;
    createdBy?: ActorId; // e.g. the model actor, while the operation actor is the dispatcher
    inputRevisionIds?: RevisionId[]; // everything the proposer saw (may exceed the basis)
  },
): { proposalId: ProposalId; commit: Commit } {
  const opId = newOperationId();
  const when = t(ctx);
  const proposal: Proposal = {
    id: newProposalId(),
    kind: opts.kind,
    status: 'open',
    basisRevisionIds: opts.basisRevisionIds,
    targetChunkIds: opts.targetChunkIds,
    payload: opts.payload,
    note: opts.note,
    createdBy: opts.createdBy ?? ctx.actorId,
    createdAt: when,
  };
  const commit = finish(
    ctx,
    {
      id: opId,
      kind: 'propose',
      at: when,
      inputs: opts.inputRevisionIds ?? opts.basisRevisionIds,
      params: { kind: opts.kind },
    },
    { proposals: [proposal] },
    opts.targetChunkIds,
  );
  return { proposalId: proposal.id, commit };
}

// Why is this proposal no longer applicable, or null if it is fresh.
export function staleReason(state: SubstrateState, p: Proposal): string | null {
  for (const ch of p.payload) {
    if (ch.op === 'revise') {
      const chunk = state.chunks.get(ch.chunkId);
      if (!chunk) return `target chunk ${ch.chunkId} no longer exists`;
      if (ch.mergeParentRevisionIds) {
        // A merge is fresh only while the head is still one of its parents;
        // past that, accepting would silently discard a newer revision.
        if (!ch.mergeParentRevisionIds.includes(chunk.currentRevisionId)) {
          return `chunk ${ch.chunkId} advanced past the merge parents`;
        }
        continue;
      }
      if (!p.basisRevisionIds.includes(chunk.currentRevisionId)) {
        return `chunk ${ch.chunkId} has moved on since the proposal's basis`;
      }
    } else if (ch.op === 'repin') {
      const occ = state.occurrences.get(ch.occurrenceId);
      if (!occ) return `occurrence ${ch.occurrenceId} no longer exists`;
      if (occ.pin === 'current') return `occurrence ${ch.occurrenceId} follows current; nothing to update`;
      if (!p.basisRevisionIds.includes(occ.pin)) {
        return `occurrence ${ch.occurrenceId} was repinned since the proposal's basis`;
      }
    } else if (ch.op === 'sever') {
      if (!state.occurrences.has(ch.occurrenceId)) return `occurrence ${ch.occurrenceId} already severed`;
    } else if (ch.op === 'place') {
      if (typeof ch.chunkId === 'string' && !state.chunks.has(ch.chunkId)) return `chunk ${ch.chunkId} no longer exists`;
      if (!state.chunks.has(ch.containerId)) return `container ${ch.containerId} no longer exists`;
    }
  }
  // Additive proposals (generation-shaped: only create/place/relate) anchor
  // their freshness to the target chunks: if every target has moved past the
  // basis, the proposal was computed against context that no longer exists.
  if (
    p.payload.length > 0 &&
    p.targetChunkIds.length > 0 &&
    p.basisRevisionIds.length > 0 &&
    p.payload.every((ch) => ch.op === 'create' || ch.op === 'place' || ch.op === 'relate')
  ) {
    const anchored = p.targetChunkIds.some((t) => {
      const chunk = state.chunks.get(t);
      return chunk !== undefined && p.basisRevisionIds.includes(chunk.currentRevisionId);
    });
    if (!anchored) return `no target chunk is still at the proposal's basis`;
  }
  return null;
}

export async function acceptProposal(
  ctx: TxCtx,
  opts: { proposalId: ProposalId },
): Promise<{ applied: boolean; reason?: string; commit: Commit; createdChunkIds: ChunkId[] }> {
  const p = ctx.state.proposals.get(opts.proposalId);
  if (!p) throw new Error(`accept: unknown proposal ${opts.proposalId}`);
  if (p.status !== 'open') throw new Error(`accept: proposal ${opts.proposalId} is ${p.status}`);

  const stale = staleReason(ctx.state, p);
  const when = t(ctx);
  if (stale) {
    const commit = finish(
      ctx,
      { id: newOperationId(), kind: 'reject', at: when, params: { result: 'superseded', reason: stale } },
      { proposalUpdates: [{ id: p.id, status: 'superseded', resolution: { by: ctx.actorId, at: when } }] },
      p.targetChunkIds,
    );
    return { applied: false, reason: stale, commit, createdChunkIds: [] };
  }

  const opId = newOperationId();
  const facts: Facts = {
    blobs: [],
    chunks: [],
    revisions: [],
    setCurrent: [],
    occurrences: [],
    occurrenceUpdates: [],
    removeOccurrences: [],
    links: [],
    derivations: [],
  };
  const temp = new Map<string, ChunkId>();
  const createdChunkIds: ChunkId[] = [];
  // Positions within one accept chain: an anchored place (after / start) sets
  // the cursor, and unanchored places follow the cursor inside its bound, so
  // a run of inserted blocks keeps its payload order wherever it lands.
  const cursor = new Map<ChunkId, { last: string | null; bound: string | null }>();
  const nextPosition = (containerId: ChunkId, after?: OccurrenceId, at?: 'start'): string => {
    let c: { last: string | null; bound: string | null };
    if (after) {
      const sibs = childOccurrences(ctx.state, containerId);
      const i = sibs.findIndex((o) => o.id === after);
      if (i < 0) throw new Error(`accept: anchor occurrence ${after} not found`);
      c = { last: sibs[i].position, bound: sibs[i + 1]?.position ?? null };
    } else if (at === 'start') {
      const sibs = childOccurrences(ctx.state, containerId);
      c = { last: null, bound: sibs.length ? sibs[0].position : null };
    } else {
      c = cursor.get(containerId) ?? {
        last: (() => {
          const sibs = childOccurrences(ctx.state, containerId);
          return sibs.length ? sibs[sibs.length - 1].position : null;
        })(),
        bound: null,
      };
    }
    const pos = keyBetween(c.last, c.bound);
    cursor.set(containerId, { last: pos, bound: c.bound });
    return pos;
  };

  for (const ch of p.payload) {
    if (ch.op === 'create') {
      const mediaType = ch.mediaType ?? MEDIA_TEXT;
      const blob = await makeBlob(mediaType, ch.text);
      const chunkId = newChunkId();
      temp.set(ch.tempId, chunkId);
      createdChunkIds.push(chunkId);
      const revision: Revision = {
        id: newRevisionId(),
        chunkId,
        blobHash: blob.hash,
        mediaType,
        parentRevisionIds: [],
        createdBy: p.createdBy, // the proposer authored the content; the accept admitted it
        createdAt: when,
        operationId: opId,
      };
      facts.blobs!.push(blob);
      facts.chunks!.push({ id: chunkId, currentRevisionId: revision.id, tombstoned: false });
      facts.revisions!.push(revision);
      if (ch.derivedFrom) {
        facts.derivations!.push({
          id: newDerivationId(),
          childChunkId: chunkId,
          sourceRevisionId: ch.derivedFrom.sourceRevisionId,
          sourceSpan: ch.derivedFrom.sourceSpan,
          via: ch.derivedFrom.via,
          operationId: opId,
        });
      }
    } else if (ch.op === 'revise') {
      const cur = currentRevision(ctx.state, ch.chunkId);
      const mediaType = ch.mediaType ?? cur.mediaType;
      const blob = await makeBlob(mediaType, ch.text);
      const revision: Revision = {
        id: newRevisionId(),
        chunkId: ch.chunkId,
        blobHash: blob.hash,
        mediaType,
        parentRevisionIds: ch.mergeParentRevisionIds ?? [cur.id],
        createdBy: p.createdBy,
        createdAt: when,
        operationId: opId,
      };
      facts.blobs!.push(blob);
      facts.revisions!.push(revision);
      facts.setCurrent!.push({ chunkId: ch.chunkId, revisionId: revision.id });
    } else if (ch.op === 'place') {
      const chunkId = typeof ch.chunkId === 'string' ? ch.chunkId : temp.get(ch.chunkId.tempId);
      if (!chunkId) throw new Error(`accept: unresolved tempId in place`);
      facts.occurrences!.push({
        id: newOccurrenceId(),
        containerId: ch.containerId,
        chunkId,
        position: nextPosition(ch.containerId, ch.after, ch.at),
        mode: ch.mode ?? 'contain',
        pin: 'current',
        watch: ch.watch ?? false,
      });
    } else if (ch.op === 'repin') {
      facts.occurrenceUpdates!.push({ id: ch.occurrenceId, pin: ch.revisionId });
    } else if (ch.op === 'relate') {
      facts.links!.push({
        id: newLinkId(),
        fromChunkId: ch.fromChunkId,
        toChunkId: ch.toChunkId,
        toExternal: ch.toExternal,
        role: ch.role,
        operationId: opId,
      });
    } else if (ch.op === 'sever') {
      facts.removeOccurrences!.push(ch.occurrenceId);
    }
  }
  facts.proposalUpdates = [
    { id: p.id, status: 'accepted', resolution: { by: ctx.actorId, at: when, operationId: opId } },
  ];
  const commit = finish(
    ctx,
    {
      id: opId,
      kind: 'accept',
      at: when,
      inputs: p.basisRevisionIds,
      outputs: facts.revisions!.map((r) => r.id),
      params: { proposalId: p.id, proposalKind: p.kind },
    },
    facts,
    p.targetChunkIds,
  );
  return { applied: true, commit, createdChunkIds };
}

// A proposal overtaken by events: recorded, sedimentary, never silently gone.
export function supersedeProposal(ctx: TxCtx, opts: { proposalId: ProposalId; reason: string }): Commit {
  const p = ctx.state.proposals.get(opts.proposalId);
  if (!p) throw new Error(`supersede: unknown proposal ${opts.proposalId}`);
  if (p.status !== 'open') throw new Error(`supersede: proposal ${opts.proposalId} is ${p.status}`);
  const when = t(ctx);
  return finish(
    ctx,
    { id: newOperationId(), kind: 'reject', at: when, params: { result: 'superseded', reason: opts.reason } },
    { proposalUpdates: [{ id: p.id, status: 'superseded', resolution: { by: ctx.actorId, at: when } }] },
    p.targetChunkIds,
  );
}

export function rejectProposal(ctx: TxCtx, opts: { proposalId: ProposalId }): Commit {
  const p = ctx.state.proposals.get(opts.proposalId);
  if (!p) throw new Error(`reject: unknown proposal ${opts.proposalId}`);
  if (p.status !== 'open') throw new Error(`reject: proposal ${opts.proposalId} is ${p.status}`);
  const when = t(ctx);
  return finish(
    ctx,
    { id: newOperationId(), kind: 'reject', at: when },
    { proposalUpdates: [{ id: p.id, status: 'rejected', resolution: { by: ctx.actorId, at: when } }] },
    p.targetChunkIds,
  );
}

// ── watched transclusion scanning ────────────────────────────────────────────

// Detect watched, pinned occurrences whose source has moved on and raise
// source-update proposals. Stale open updates are superseded, then replaced.
export function scanWatchedSources(ctx: TxCtx): ProposalId[] {
  const raised: ProposalId[] = [];
  const openUpdates = new Map<OccurrenceId, Proposal>();
  for (const p of ctx.state.proposals.values()) {
    if (p.status !== 'open' || p.kind !== 'source-update') continue;
    const repin = p.payload.find((c) => c.op === 'repin');
    if (repin && repin.op === 'repin') openUpdates.set(repin.occurrenceId, p);
  }
  for (const occ of [...ctx.state.occurrences.values()]) {
    if (occ.mode !== 'transclude' || !occ.watch || occ.pin === 'current') continue;
    const head = currentRevision(ctx.state, occ.chunkId);
    if (head.id === occ.pin) continue;
    const existing = openUpdates.get(occ.id);
    if (existing) {
      const repin = existing.payload.find((c) => c.op === 'repin');
      if (repin && repin.op === 'repin' && repin.revisionId === head.id) continue; // still accurate
      supersedeProposal(ctx, { proposalId: existing.id, reason: 'newer source revision' });
    }
    const { proposalId } = propose(ctx, {
      kind: 'source-update',
      basisRevisionIds: [occ.pin],
      targetChunkIds: [occ.containerId],
      payload: [{ op: 'repin', occurrenceId: occ.id, revisionId: head.id }],
      note: `watched source ${occ.chunkId} advanced to a new revision`,
    });
    raised.push(proposalId);
  }
  return raised;
}
