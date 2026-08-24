// The canonical pipeline: select gathers candidates with their roles, reduce
// compiles a bounded structured context (provenance is never erased into a bare
// string), and generation returns as a proposal — model output never applies
// itself.

import {
  childOccurrences,
  occurrenceRevision,
  occurrencesOfChunk,
  renderRevision,
  isComposite,
  type WorkspaceGraph,
} from './state';
import { propose, type TxCtx } from './tx';
import type {
  ActorId,
  ChunkId,
  ContextRevisionPrecondition,
  ContextStructurePrecondition,
  Occurrence,
  OccurrenceId,
  OccurrencePrecondition,
  ProposalId,
  ProducerRef,
  RevisionId,
} from './types';
import { isCompositeMediaType, MEDIA_MARKDOWN } from './types';

export type ContextRole = 'focus' | 'child' | 'parent' | 'sibling' | 'search';

export interface ContextItem {
  chunkId: ChunkId;
  revisionId: RevisionId;
  text: string;
  role: ContextRole;
  occurrenceId?: OccurrenceId;
  // Exact immutable revisions that contributed to this rendered item. Pinned
  // descendants remain provenance inputs without falsely following the head.
  dependencies: ContextRevisionPrecondition[];
}

const ROLE_ORDER: ContextRole[] = ['focus', 'child', 'parent', 'sibling', 'search'];

function revisionDependencies(
  state: WorkspaceGraph,
  chunkId: ChunkId,
  rootRevisionId: RevisionId,
  rootFollowsCurrent: boolean,
  rendered: boolean,
): ContextRevisionPrecondition[] {
  const dependencies = new Map<RevisionId, ContextRevisionPrecondition>();
  const add = (revisionId: RevisionId, followsCurrent: boolean): void => {
    const revision = state.revisions.get(revisionId);
    if (!revision) return;
    const chunk = state.chunks.get(revision.chunkId);
    if (!chunk) return;
    const existing = dependencies.get(revisionId);
    dependencies.set(revisionId, {
      chunkId: revision.chunkId,
      revisionId,
      followsCurrent: followsCurrent || existing?.followsCurrent === true,
      redacted: Boolean(revision.redacted),
      chunkTombstoned: chunk.tombstoned,
    });
  };
  const visitRendered = (
    currentChunkId: ChunkId,
    revisionId: RevisionId,
    followsCurrent: boolean,
    seen: Set<ChunkId>,
  ): void => {
    if (seen.has(currentChunkId)) return;
    seen.add(currentChunkId);
    const revision = state.revisions.get(revisionId);
    if (!revision || revision.chunkId !== currentChunkId) return;
    add(revision.id, followsCurrent);
    if (!revision.redacted && isCompositeMediaType(revision.mediaType)) {
      for (const occurrence of childOccurrences(state, currentChunkId)) {
        const effective = occurrenceRevision(state, occurrence);
        visitRendered(occurrence.chunkId, effective.id, occurrence.pin === 'current', seen);
      }
    }
    seen.delete(currentChunkId);
  };
  if (rendered) visitRendered(chunkId, rootRevisionId, rootFollowsCurrent, new Set());
  else add(rootRevisionId, rootFollowsCurrent);
  return [...dependencies.values()];
}

// Gather the focus, its children, its containers and their other children, and
// any externally supplied hits (e.g. index search results). Inspectable: every
// item says why it is here.
export function select(state: WorkspaceGraph, focusId: ChunkId, searchHits: ChunkId[] = []): ContextItem[] {
  const picked = new Map<ChunkId, ContextItem>();
  const add = (
    chunkId: ChunkId,
    role: ContextRole,
    opts: {
      revisionId?: RevisionId;
      followsCurrent?: boolean;
      occurrenceId?: OccurrenceId;
      text?: string;
    } = {},
  ) => {
    const chunk = state.chunks.get(chunkId);
    if (!chunk || chunk.tombstoned) return;
    const existing = picked.get(chunkId);
    if (existing && ROLE_ORDER.indexOf(existing.role) <= ROLE_ORDER.indexOf(role)) return;
    const revisionId = opts.revisionId ?? chunk.currentRevisionId;
    const followsCurrent = opts.followsCurrent ?? true;
    const hasExplicitText = Object.prototype.hasOwnProperty.call(opts, 'text');
    picked.set(chunkId, {
      chunkId,
      revisionId,
      text: hasExplicitText ? opts.text! : renderRevision(state, revisionId),
      role,
      occurrenceId: opts.occurrenceId,
      dependencies: revisionDependencies(state, chunkId, revisionId, followsCurrent, role !== 'parent'),
    });
  };
  add(focusId, 'focus');
  for (const occ of childOccurrences(state, focusId)) {
    const rev = occurrenceRevision(state, occ);
    add(occ.chunkId, 'child', {
      revisionId: rev.id,
      followsCurrent: occ.pin === 'current',
      occurrenceId: occ.id,
    });
  }
  for (const occ of occurrencesOfChunk(state, focusId)) {
    add(occ.containerId, 'parent', { text: '' });
    for (const sib of childOccurrences(state, occ.containerId)) {
      if (sib.chunkId !== focusId) {
        const revision = occurrenceRevision(state, sib);
        add(sib.chunkId, 'sibling', {
          revisionId: revision.id,
          followsCurrent: sib.pin === 'current',
          occurrenceId: sib.id,
        });
      }
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

// Keep whole items in role-priority order until the budget is spent. The focus
// is never dropped: if it alone exceeds the budget it is truncated to fit.
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
      if (item.role === 'focus' && chars < budget) {
        const clipped = item.text.slice(0, budget - chars);
        kept.push({ ...item, text: clipped });
        chars += clipped.length;
        continue;
      }
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
export interface CompletionOutput {
  text: string;
  producer?: ProducerRef;
}

export type Completer = (
  context: ReducedContext,
  instruction: string,
) => Promise<string | CompletionOutput>;

const occurrencePrecondition = (occurrence: Occurrence): OccurrencePrecondition => ({
  id: occurrence.id,
  containerId: occurrence.containerId,
  chunkId: occurrence.chunkId,
  position: occurrence.position,
  mode: occurrence.mode,
  pin: occurrence.pin,
  watch: occurrence.watch,
});

function contextStructurePrecondition(
  state: WorkspaceGraph,
  context: ReducedContext,
): ContextStructurePrecondition {
  const containerIds = new Set<ChunkId>();
  const collectRenderedContainers = (
    chunkId: ChunkId,
    revisionId: RevisionId,
    nested: boolean,
    seen = new Set<ChunkId>(),
  ): void => {
    const revision = state.revisions.get(revisionId);
    if (
      seen.has(chunkId) ||
      !revision ||
      revision.chunkId !== chunkId ||
      revision.redacted ||
      !isCompositeMediaType(revision.mediaType)
    ) return;
    seen.add(chunkId);
    containerIds.add(chunkId);
    if (nested) {
      for (const occurrence of childOccurrences(state, chunkId)) {
        const effective = occurrenceRevision(state, occurrence);
        collectRenderedContainers(occurrence.chunkId, effective.id, true, seen);
      }
    }
  };
  for (const item of context.items) {
    // Parent text is intentionally omitted from the prompt, but its immediate
    // children determine why siblings were selected. Other composite items are
    // rendered recursively, so every nested container is a dependency.
    collectRenderedContainers(item.chunkId, item.revisionId, item.role !== 'parent');
  }
  const containers = [...containerIds]
    .sort()
    .map((containerId) => ({
      containerId,
      occurrences: childOccurrences(state, containerId).map(occurrencePrecondition),
    }));
  const placements = [...new Set(context.items.map((item) => item.chunkId))]
    .sort()
    .map((chunkId) => ({
      chunkId,
      occurrences: occurrencesOfChunk(state, chunkId)
        .sort((a, b) => a.id.localeCompare(b.id))
        .map(occurrencePrecondition),
    }));
  return { containers, placements };
}

// Dispatch: select → reduce → complete → propose. The proposal's creator is the
// model actor; the operation's actor is whoever dispatched.
export async function generateProposal(
  ctx: TxCtx,
  opts: {
    focusChunkId: ChunkId;
    instruction: string;
    complete: Completer;
    modelActorId: ActorId;
    searchHits?: ChunkId[];
    budget?: number;
  },
): Promise<{ proposalId: ProposalId; context: ReducedContext }> {
  const items = select(ctx.state, opts.focusChunkId, opts.searchHits ?? []);
  const context = reduce(items, opts.budget);
  const freshnessStructure = contextStructurePrecondition(ctx.state, context);
  const dependencyMap = new Map<RevisionId, ContextRevisionPrecondition>();
  for (const dependency of context.items.flatMap((item) => item.dependencies)) {
    const existing = dependencyMap.get(dependency.revisionId);
    dependencyMap.set(dependency.revisionId, {
      ...dependency,
      followsCurrent: dependency.followsCurrent || existing?.followsCurrent === true,
    });
  }
  const freshnessRevisionStates = [...dependencyMap.values()];
  // Capture the exact focus revision before yielding to a provider. The
  // provider may be slow while humans keep editing; its basis and derivation
  // must describe what it actually saw, never a head reread after completion.
  const focusItem = context.items.find((item) => item.chunkId === opts.focusChunkId && item.role === 'focus');
  if (!focusItem) throw new Error(`dispatch: focus chunk ${opts.focusChunkId} was not selected`);
  const focusRev = ctx.state.revisions.get(focusItem.revisionId);
  if (!focusRev) throw new Error(`dispatch: focus revision ${focusItem.revisionId} no longer exists`);
  const focusWasComposite = isComposite(ctx.state, opts.focusChunkId);
  const focusHome = focusWasComposite ? undefined : occurrencesOfChunk(ctx.state, opts.focusChunkId)[0];
  const completed = await opts.complete(context, opts.instruction);
  const text = typeof completed === 'string' ? completed : completed.text;
  const producer = typeof completed === 'string' ? undefined : completed.producer;
  if (!text.trim()) throw new Error('dispatch: collaborator returned empty text');

  const payload: import('./types').ProposedChange[] = [
    {
      op: 'create',
      tempId: 'generated',
      text,
      mediaType: MEDIA_MARKDOWN,
      derivedFrom: { sourceRevisionId: focusRev.id, via: 'generate' },
    },
  ];
  if (focusWasComposite) {
    payload.push({ op: 'place', containerId: opts.focusChunkId, chunkId: { tempId: 'generated' } });
  } else {
    if (focusHome) {
      payload.push({
        op: 'place',
        containerId: focusHome.containerId,
        chunkId: { tempId: 'generated' },
        after: focusHome.id,
      });
    }
  }

  const { proposalId } = propose(ctx, {
    kind: 'generation',
    basisRevisionIds: [focusRev.id],
    targetChunkIds: [opts.focusChunkId],
    payload,
    note: opts.instruction,
    createdBy: opts.modelActorId,
    producer,
    // The operation records everything the generator saw, not just the anchor.
    inputRevisionIds: freshnessRevisionStates.map((dependency) => dependency.revisionId),
    freshnessRevisionIds: freshnessRevisionStates
      .filter((dependency) => dependency.followsCurrent)
      .map((dependency) => dependency.revisionId),
    freshnessRevisionStates,
    freshnessStructure,
  });
  return { proposalId, context };
}
