// The canonical pipeline: select gathers candidates with their roles, reduce
// compiles a bounded structured context (provenance is never erased into a bare
// string), and generation returns as a proposal — model output never applies
// itself.

import {
  childOccurrences,
  currentRevision,
  occurrenceRevision,
  occurrencesOfChunk,
  renderChunk,
  isComposite,
  type SubstrateState,
} from './state';
import { propose, type TxCtx } from './tx';
import type { ActorId, ChunkId, ProposalId, RevisionId } from './types';
import { MEDIA_MARKDOWN } from './types';

export type ContextRole = 'focus' | 'child' | 'parent' | 'sibling' | 'search';

export interface ContextItem {
  chunkId: ChunkId;
  revisionId: RevisionId;
  text: string;
  role: ContextRole;
}

const ROLE_ORDER: ContextRole[] = ['focus', 'child', 'parent', 'sibling', 'search'];

// Gather the focus, its children, its containers and their other children, and
// any externally supplied hits (e.g. index search results). Inspectable: every
// item says why it is here.
export function select(state: SubstrateState, focusId: ChunkId, searchHits: ChunkId[] = []): ContextItem[] {
  const picked = new Map<ChunkId, ContextItem>();
  const add = (chunkId: ChunkId, role: ContextRole, text?: string) => {
    const chunk = state.chunks.get(chunkId);
    if (!chunk || chunk.tombstoned) return;
    const existing = picked.get(chunkId);
    if (existing && ROLE_ORDER.indexOf(existing.role) <= ROLE_ORDER.indexOf(role)) return;
    picked.set(chunkId, {
      chunkId,
      revisionId: chunk.currentRevisionId,
      text: text ?? renderChunk(state, chunkId),
      role,
    });
  };
  add(focusId, 'focus');
  for (const occ of childOccurrences(state, focusId)) {
    const rev = occurrenceRevision(state, occ);
    add(occ.chunkId, 'child', rev.mediaType === 'application/x-substrate-composite' ? renderChunk(state, occ.chunkId) : undefined);
  }
  for (const occ of occurrencesOfChunk(state, focusId)) {
    add(occ.containerId, 'parent', '');
    for (const sib of childOccurrences(state, occ.containerId)) {
      if (sib.chunkId !== focusId) add(sib.chunkId, 'sibling');
    }
  }
  for (const hit of searchHits) add(hit, 'search');
  return [...picked.values()].sort((a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role));
}

export interface ReducedContext {
  items: ContextItem[];
  chars: number;
  dropped: number;
}

// Keep whole items in role-priority order until the budget is spent.
export function reduce(items: ContextItem[], budget = 6000): ReducedContext {
  const kept: ContextItem[] = [];
  let chars = 0;
  let dropped = 0;
  for (const item of items) {
    if (item.text.length === 0) {
      kept.push(item);
      continue;
    }
    if (chars + item.text.length > budget) {
      dropped++;
      continue;
    }
    kept.push(item);
    chars += item.text.length;
  }
  return { items: kept, chars, dropped };
}

// The generation seam: any provider (or a human) that turns reduced context and
// an instruction into text. Replaceable; the kernel only sees the signature.
export type Completer = (context: ReducedContext, instruction: string) => Promise<string>;

export const stubCompleter: Completer = async (context, instruction) => {
  const focus = context.items.find((i) => i.role === 'focus');
  return (
    `(stub) In response to "${instruction}" — a real model driver plugs in behind the Completer seam. ` +
    `It saw ${context.items.length} context items (${context.chars} chars), focused on ${focus?.chunkId ?? 'nothing'}.`
  );
};

// Dispatch: select → reduce → complete → propose. The proposal's creator is the
// model actor; the operation's actor is whoever dispatched.
export async function generateProposal(
  ctx: TxCtx,
  opts: {
    focusChunkId: ChunkId;
    instruction: string;
    complete?: Completer;
    modelActorId?: ActorId;
    searchHits?: ChunkId[];
    budget?: number;
  },
): Promise<{ proposalId: ProposalId; context: ReducedContext }> {
  const items = select(ctx.state, opts.focusChunkId, opts.searchHits ?? []);
  const context = reduce(items, opts.budget);
  const text = await (opts.complete ?? stubCompleter)(context, opts.instruction);
  const focusRev = currentRevision(ctx.state, opts.focusChunkId);

  const payload: import('./types').ProposedChange[] = [
    {
      op: 'create',
      tempId: 'generated',
      text,
      mediaType: MEDIA_MARKDOWN,
      derivedFrom: { sourceRevisionId: focusRev.id, via: 'generate' },
    },
  ];
  if (isComposite(ctx.state, opts.focusChunkId)) {
    payload.push({ op: 'place', containerId: opts.focusChunkId, chunkId: { tempId: 'generated' } });
  } else {
    const home = occurrencesOfChunk(ctx.state, opts.focusChunkId)[0];
    if (home) payload.push({ op: 'place', containerId: home.containerId, chunkId: { tempId: 'generated' }, after: home.id });
  }

  const { proposalId } = propose(ctx, {
    kind: 'generation',
    basisRevisionIds: [focusRev.id],
    targetChunkIds: [opts.focusChunkId],
    payload,
    note: opts.instruction,
    createdBy: opts.modelActorId ?? 'agent:stub',
  });
  return { proposalId, context };
}
