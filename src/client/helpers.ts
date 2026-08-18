// View-side derivations over kernel state. Read-only: views never mutate truth.

import {
  childOccurrences,
  currentRevision,
  isComposite,
  occurrenceRevision,
  renderChunk,
  type SubstrateState,
} from '../kernel/state';
import type { ChunkId, Occurrence, Proposal, Revision } from '../kernel/types';
import type { BindingInfo } from './useSubstrate';

// A leaf block as the Star renders it: transclusions arrive read-only.
export interface LeafBlock {
  occurrence: Occurrence;
  chunkId: ChunkId;
  revision: Revision;
  text: string;
  depth: number;
  transcluded: boolean;
}

export function leafBlocks(state: SubstrateState, containerId: ChunkId, depth = 0, seen = new Set<ChunkId>()): LeafBlock[] {
  if (seen.has(containerId)) return [];
  seen.add(containerId);
  const out: LeafBlock[] = [];
  for (const occ of childOccurrences(state, containerId)) {
    const rev = occurrenceRevision(state, occ);
    const transcluded = occ.mode === 'transclude';
    if (rev.mediaType === 'application/x-substrate-composite') {
      out.push(...leafBlocks(state, occ.chunkId, depth + 1, seen).map((b) => ({ ...b, transcluded: b.transcluded || transcluded })));
    } else {
      out.push({
        occurrence: occ,
        chunkId: occ.chunkId,
        revision: rev,
        text: rev.redacted ? '[redacted]' : (state.blobs.get(rev.blobHash)?.text ?? ''),
        depth,
        transcluded,
      });
    }
  }
  seen.delete(containerId);
  return out;
}

export function labelOf(state: SubstrateState, bindings: BindingInfo[], chunkId: ChunkId): string {
  const bound = bindings.find((b) => b.docChunkId === chunkId);
  if (bound) return bound.relPath;
  const text = renderChunk(state, chunkId).trim();
  const firstLine = text.split('\n')[0].replace(/^#+\s*/, '');
  return firstLine.length > 48 ? `${firstLine.slice(0, 48)}…` : firstLine || chunkId.slice(0, 12);
}

// The nebula's stars: bound docs plus unbound root composites (e.g. accepted
// generations that grew into documents).
export function docList(state: SubstrateState, bindings: BindingInfo[]): ChunkId[] {
  const docs = new Set<ChunkId>();
  for (const b of bindings) if (state.chunks.get(b.docChunkId) && !state.chunks.get(b.docChunkId)!.tombstoned) docs.add(b.docChunkId);
  const contained = new Set<ChunkId>();
  for (const occ of state.occurrences.values()) contained.add(occ.chunkId);
  for (const chunk of state.chunks.values()) {
    if (chunk.tombstoned || docs.has(chunk.id) || contained.has(chunk.id)) continue;
    if (isComposite(state, chunk.id)) docs.add(chunk.id);
  }
  return [...docs];
}

export function revisionCount(state: SubstrateState, chunkId: ChunkId): number {
  let n = 0;
  for (const r of state.revisions.values()) if (r.chunkId === chunkId) n++;
  return n;
}

// Open proposals touching this doc or anything rendered inside it.
export function proposalsForDoc(state: SubstrateState, docId: ChunkId): Proposal[] {
  const scope = new Set<ChunkId>([docId, ...leafBlocks(state, docId).map((b) => b.chunkId)]);
  const out: Proposal[] = [];
  for (const p of state.proposals.values()) {
    if (p.status !== 'open') continue;
    if (p.targetChunkIds.some((t) => scope.has(t))) out.push(p);
  }
  return out.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

export function currentText(state: SubstrateState, chunkId: ChunkId): string {
  const rev = currentRevision(state, chunkId);
  return rev.redacted ? '[redacted]' : (state.blobs.get(rev.blobHash)?.text ?? '');
}
