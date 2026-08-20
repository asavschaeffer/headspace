// View-side derivations over kernel state. Read-only: views never mutate truth.

import {
  childOccurrences,
  currentRevision,
  isComposite,
  occurrencesOfChunk,
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
    if (rev.mediaType === 'application/x-substrate-composite' && !transcluded) {
      out.push(...leafBlocks(state, occ.chunkId, depth + 1, seen));
    } else if (rev.mediaType === 'application/x-substrate-composite') {
      // A transcluded composite renders as ONE read-only block: descending
      // would expose the source's internal occurrences to sever/move —
      // authority the transclusion does not grant (wiki/deep-fates.md).
      out.push({
        occurrence: occ,
        chunkId: occ.chunkId,
        revision: rev,
        text: renderChunk(state, occ.chunkId),
        depth,
        transcluded: true,
      });
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

// Where should the UI land when following a link to this chunk? Leaf chunks
// open through their nearest container, so the star never opens empty.
export function focusTarget(state: SubstrateState, chunkId: ChunkId): ChunkId {
  let current = chunkId;
  const seen = new Set<ChunkId>();
  while (!seen.has(current)) {
    seen.add(current);
    const chunk = state.chunks.get(current);
    if (!chunk) return chunkId;
    if (isComposite(state, current)) return current;
    const home = occurrencesOfChunk(state, current)[0];
    if (!home) return current;
    current = home.containerId;
  }
  return current;
}

// All containers reachable upward from a chunk (for illumination: a match
// deep inside nested composites still lights its document).
export function ancestorContainers(state: SubstrateState, chunkId: ChunkId): ChunkId[] {
  const out: ChunkId[] = [];
  const seen = new Set<ChunkId>([chunkId]);
  const frontier = [chunkId];
  while (frontier.length) {
    const cur = frontier.pop()!;
    for (const occ of occurrencesOfChunk(state, cur)) {
      if (seen.has(occ.containerId)) continue;
      seen.add(occ.containerId);
      out.push(occ.containerId);
      frontier.push(occ.containerId);
    }
  }
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
