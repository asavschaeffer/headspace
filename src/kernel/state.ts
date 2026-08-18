import type {
  BlobHash,
  Chunk,
  ChunkId,
  Commit,
  Occurrence,
  OccurrenceId,
  Proposal,
  Revision,
  RevisionId,
} from './types';
import { MEDIA_COMPOSITE } from './types';

// The materialized truth: what the log says, folded into queryable maps.
// The log (commits) is authoritative; this state is its deterministic product.

export interface SubstrateState {
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

export function emptyState(): SubstrateState {
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

const fail = (msg: string): never => {
  throw new InvariantError(msg);
};

// Would placing `chunkId` inside `containerId` close a containment cycle?
// Placement adds the edge container -> chunk; a cycle exists iff container is
// reachable downward from chunk (or they are the same identity). `extraEdges`
// lets a commit be validated with its own not-yet-applied occurrences included.
export function wouldCreateCycle(
  state: SubstrateState,
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

// Validate then fold one commit into state. Mutates in place; callers that need
// react-style change detection bump their own version counter per commit.
export function applyCommit(state: SubstrateState, commit: Commit): void {
  const f = commit.facts;
  const newChunks = new Set((f.chunks ?? []).map((c) => c.id));

  for (const r of f.revisions ?? []) {
    if (state.revisions.has(r.id)) fail(`revision ${r.id} already exists; revisions are immutable`);
    if (!state.chunks.has(r.chunkId) && !newChunks.has(r.chunkId)) fail(`revision ${r.id} for unknown chunk ${r.chunkId}`);
    const hasBlob = state.blobs.has(r.blobHash) || (f.blobs ?? []).some((b) => b.hash === r.blobHash);
    if (!hasBlob) fail(`revision ${r.id} points at missing blob ${r.blobHash}`);
    for (const p of r.parentRevisionIds) {
      if (!state.revisions.has(p)) fail(`revision ${r.id} has unknown parent ${p}`);
    }
  }
  const newRevisions = new Map((f.revisions ?? []).map((r) => [r.id, r]));
  for (const c of f.chunks ?? []) {
    if (state.chunks.has(c.id)) fail(`chunk ${c.id} already exists`);
    const rev = newRevisions.get(c.currentRevisionId);
    if (!rev || rev.chunkId !== c.id) fail(`chunk ${c.id} current revision must be its own new revision`);
  }
  for (const s of f.setCurrent ?? []) {
    const rev = newRevisions.get(s.revisionId) ?? state.revisions.get(s.revisionId);
    if (!rev) fail(`setCurrent: unknown revision ${s.revisionId}`);
    if (rev!.chunkId !== s.chunkId) fail(`setCurrent: revision ${s.revisionId} belongs to ${rev!.chunkId}, not ${s.chunkId}`);
    if (!state.chunks.has(s.chunkId) && !newChunks.has(s.chunkId)) fail(`setCurrent: unknown chunk ${s.chunkId}`);
  }
  const newEdges = (f.occurrences ?? []).map((o) => ({ containerId: o.containerId, chunkId: o.chunkId }));
  for (const [i, o] of (f.occurrences ?? []).entries()) {
    if (state.occurrences.has(o.id)) fail(`occurrence ${o.id} already exists`);
    if (!state.chunks.has(o.containerId) && !newChunks.has(o.containerId)) fail(`occurrence container ${o.containerId} unknown`);
    if (!state.chunks.has(o.chunkId) && !newChunks.has(o.chunkId)) fail(`occurrence chunk ${o.chunkId} unknown`);
    if (!o.position) fail(`occurrence ${o.id} missing position`);
    if (wouldCreateCycle(state, o.containerId, o.chunkId, newEdges.slice(0, i))) {
      fail(`occurrence ${o.id} would create a containment cycle (${o.containerId} <- ${o.chunkId})`);
    }
  }
  for (const u of f.occurrenceUpdates ?? []) {
    if (!state.occurrences.has(u.id)) fail(`occurrenceUpdate: unknown occurrence ${u.id}`);
  }
  for (const id of f.removeOccurrences ?? []) {
    if (!state.occurrences.has(id)) fail(`removeOccurrences: unknown occurrence ${id}`);
  }
  for (const u of f.proposalUpdates ?? []) {
    if (!state.proposals.has(u.id)) fail(`proposalUpdate: unknown proposal ${u.id}`);
  }

  // All validated — fold.
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
  for (const id of f.tombstone ?? []) {
    const c = state.chunks.get(id) ?? fail(`tombstone: unknown chunk ${id}`);
    (c as Chunk).tombstoned = true;
  }
  for (const id of f.redactRevisions ?? []) {
    const r = state.revisions.get(id) ?? fail(`redact: unknown revision ${id}`);
    state.revisions.set(id, { ...(r as Revision), redacted: true });
  }
  state.operations.set(commit.operation.id, commit.operation);
  state.head = commit.id;
  state.commitCount++;
}

export function materialize(commits: Iterable<Commit>): SubstrateState {
  const state = emptyState();
  for (const c of commits) applyCommit(state, c);
  return state;
}

// ── accessors ────────────────────────────────────────────────────────────────

export function currentRevision(state: SubstrateState, chunkId: ChunkId): Revision {
  const chunk = state.chunks.get(chunkId) ?? fail(`unknown chunk ${chunkId}`);
  return state.revisions.get((chunk as Chunk).currentRevisionId) ?? fail(`chunk ${chunkId} head revision missing`);
}

export function revisionText(state: SubstrateState, revisionId: RevisionId): string {
  const rev = state.revisions.get(revisionId) ?? fail(`unknown revision ${revisionId}`);
  if ((rev as Revision).redacted) return '[redacted]';
  const blob = state.blobs.get((rev as Revision).blobHash) ?? fail(`revision ${revisionId} blob missing`);
  return blob.text;
}

export function childOccurrences(state: SubstrateState, containerId: ChunkId): Occurrence[] {
  const out: Occurrence[] = [];
  for (const occ of state.occurrences.values()) if (occ.containerId === containerId) out.push(occ);
  return out.sort((a, b) => (a.position < b.position ? -1 : a.position > b.position ? 1 : 0));
}

export function occurrencesOfChunk(state: SubstrateState, chunkId: ChunkId): Occurrence[] {
  const out: Occurrence[] = [];
  for (const occ of state.occurrences.values()) if (occ.chunkId === chunkId) out.push(occ);
  return out;
}

// The revision an occurrence renders: its pin, or the chunk's current head.
export function occurrenceRevision(state: SubstrateState, occ: Occurrence): Revision {
  if (occ.pin === 'current') return currentRevision(state, occ.chunkId);
  return (state.revisions.get(occ.pin) as Revision) ?? fail(`occurrence ${occ.id} pinned to missing revision ${occ.pin}`);
}

export function isComposite(state: SubstrateState, chunkId: ChunkId): boolean {
  return currentRevision(state, chunkId).mediaType === MEDIA_COMPOSITE;
}

// A composite blob's text is JSON: { "join": string }. The separator is part of
// how the content reads (content), while ordering stays in occurrences
// (arrangement) — so rearranging never edits content, but rendering is exact.
export function compositeJoin(state: SubstrateState, rev: Revision): string {
  try {
    const parsed = JSON.parse(state.blobs.get(rev.blobHash)?.text ?? '');
    if (typeof parsed?.join === 'string') return parsed.join;
  } catch {
    /* fall through to default */
  }
  return '\n\n';
}

// Reader's text of a chunk: leaf text, or children joined in occurrence order.
export function renderChunk(state: SubstrateState, chunkId: ChunkId, seen: Set<ChunkId> = new Set()): string {
  if (seen.has(chunkId)) return `[cycle: ${chunkId}]`;
  seen.add(chunkId);
  const rev = currentRevision(state, chunkId);
  if (rev.mediaType !== MEDIA_COMPOSITE) {
    seen.delete(chunkId);
    return revisionText(state, rev.id);
  }
  const parts = childOccurrences(state, chunkId).map((occ) => {
    const r = occurrenceRevision(state, occ);
    if (r.mediaType === MEDIA_COMPOSITE) return renderChunk(state, occ.chunkId, seen);
    return (r as Revision).redacted ? '[redacted]' : (state.blobs.get(r.blobHash)?.text ?? '');
  });
  seen.delete(chunkId);
  return parts.join(compositeJoin(state, rev));
}

export function openProposals(state: SubstrateState, targetChunkId?: ChunkId): Proposal[] {
  const out: Proposal[] = [];
  for (const p of state.proposals.values()) {
    if (p.status !== 'open') continue;
    if (targetChunkId && !p.targetChunkIds.includes(targetChunkId)) continue;
    out.push(p);
  }
  return out.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}
