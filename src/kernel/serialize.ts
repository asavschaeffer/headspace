// State <-> JSON. Used by the disk snapshot, the dev-server API, and the
// client loader, so all three stay one representation.

import { emptyState, type SubstrateState } from './state';
import type {
  Blob,
  Chunk,
  Derivation,
  Link,
  Occurrence,
  Operation,
  Proposal,
  Revision,
} from './types';

export interface SerializedState {
  chunks: Chunk[];
  revisions: Revision[];
  blobs: Blob[];
  occurrences: Occurrence[];
  links: Link[];
  derivations: Derivation[];
  proposals: Proposal[];
  operations: Operation[];
  head: string | null;
  commitCount: number;
}

export function serializeState(state: SubstrateState): SerializedState {
  return {
    chunks: [...state.chunks.values()],
    revisions: [...state.revisions.values()],
    blobs: [...state.blobs.values()],
    occurrences: [...state.occurrences.values()],
    links: [...state.links.values()],
    derivations: [...state.derivations.values()],
    proposals: [...state.proposals.values()],
    operations: [...state.operations.values()],
    head: state.head,
    commitCount: state.commitCount,
  };
}

export function deserializeState(s: SerializedState): SubstrateState {
  const state = emptyState();
  for (const c of s.chunks) state.chunks.set(c.id, { ...c });
  for (const r of s.revisions) state.revisions.set(r.id, r);
  for (const b of s.blobs) state.blobs.set(b.hash, b);
  for (const o of s.occurrences) state.occurrences.set(o.id, { ...o });
  for (const l of s.links) state.links.set(l.id, l);
  for (const d of s.derivations) state.derivations.set(d.id, d);
  for (const p of s.proposals) state.proposals.set(p.id, { ...p });
  for (const op of s.operations) state.operations.set(op.id, op);
  state.head = s.head;
  state.commitCount = s.commitCount;
  return state;
}
