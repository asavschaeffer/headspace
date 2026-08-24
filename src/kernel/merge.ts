// Three-way merge for leaf text: line-level LCS diff, diff3 against the common
// ancestor, and the result raised as a merge proposal — a merge combines two
// authored sides, so it never applies itself. Never silent loss: a conflicted
// hunk carries both sides verbatim, and resolution is authorship.

import { revisionText, type WorkspaceGraph } from './state';
import { propose, type TxCtx } from './tx';
import type { ChunkId, ProposalId, RevisionId } from './types';

// '' is zero lines, so empty-base merges don't invent a phantom blank line.
const splitLines = (s: string): string[] => (s === '' ? [] : s.split('\n'));

// ── line diff ────────────────────────────────────────────────────────────────

// A replacement region: aLines were replaced by bLines. Either side may be
// empty (pure insert/delete). Lines outside hunks match 1:1 between a and b.
export interface DiffHunk {
  aStart: number;
  aLines: string[];
  bStart: number;
  bLines: string[];
}

// Matched line pairs [ai, bi] in order, by LCS. O(n*m); leaf text is small.
function lcsMatches(a: string[], b: string[]): [number, number][] {
  const n = a.length;
  const m = b.length;
  const w = m + 1;
  const dp = new Uint32Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * w + j] =
        a[i] === b[j] ? dp[(i + 1) * w + j + 1] + 1 : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
    }
  }
  const out: [number, number][] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push([i, j]);
      i++;
      j++;
    } else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) i++;
    else j++;
  }
  return out;
}

export function diffLines(aText: string, bText: string): DiffHunk[] {
  const a = splitLines(aText);
  const b = splitLines(bText);
  const hunks: DiffHunk[] = [];
  let ai = 0;
  let bi = 0;
  // Sentinel match at the ends flushes the trailing hunk.
  for (const [ma, mb] of [...lcsMatches(a, b), [a.length, b.length] as [number, number]]) {
    if (ma > ai || mb > bi) {
      hunks.push({ aStart: ai, aLines: a.slice(ai, ma), bStart: bi, bLines: b.slice(bi, mb) });
    }
    ai = ma + 1;
    bi = mb + 1;
  }
  return hunks;
}

// ── diff3 ────────────────────────────────────────────────────────────────────

export interface Diff3Conflict {
  baseLines: string[];
  oursLines: string[];
  theirsLines: string[];
}

export interface Diff3Result {
  clean: boolean;
  merged: string;
  conflicts: Diff3Conflict[];
}

export function diff3(base: string, ours: string, theirs: string): Diff3Result {
  const baseLines = splitLines(base);
  const sideLines = [splitLines(ours), splitLines(theirs)];
  interface Hunk {
    side: 0 | 1; // 0 = ours, 1 = theirs
    baseStart: number;
    baseLen: number;
    sideStart: number;
    sideLen: number;
  }
  const hunks: Hunk[] = [];
  for (const side of [0, 1] as const) {
    for (const h of diffLines(base, side === 0 ? ours : theirs)) {
      hunks.push({ side, baseStart: h.aStart, baseLen: h.aLines.length, sideStart: h.bStart, sideLen: h.bLines.length });
    }
  }
  hunks.sort((x, y) => x.baseStart - y.baseStart || x.side - y.side);

  const out: string[] = [];
  const conflicts: Diff3Conflict[] = [];
  let basePos = 0;
  let i = 0;
  while (i < hunks.length) {
    // Group hunks whose base regions touch or overlap; like diff3, adjacent
    // edits with no stable line between them contest the same region.
    const lo = hunks[i].baseStart;
    let hi = lo + hunks[i].baseLen;
    let j = i + 1;
    while (j < hunks.length && hunks[j].baseStart <= hi) {
      hi = Math.max(hi, hunks[j].baseStart + hunks[j].baseLen);
      j++;
    }
    const group = hunks.slice(i, j);
    out.push(...baseLines.slice(basePos, lo)); // stable: all three agree

    // A side's text over base [lo, hi): its hunks extended by the aligned
    // margins. Between own hunks the side matches base 1:1, so one contiguous
    // slice of the side's lines covers the whole region.
    const sideSlice = (side: 0 | 1): string[] => {
      const own = group.filter((h) => h.side === side);
      if (own.length === 0) return baseLines.slice(lo, hi);
      const first = own[0];
      const last = own[own.length - 1];
      const from = first.sideStart - (first.baseStart - lo);
      const to = last.sideStart + last.sideLen + (hi - (last.baseStart + last.baseLen));
      return sideLines[side].slice(from, to);
    };

    const b = baseLines.slice(lo, hi);
    const o = sideSlice(0);
    const t = sideSlice(1);
    const eq = (x: string[], y: string[]) => x.length === y.length && x.every((l, k) => l === y[k]);
    if (eq(o, t)) out.push(...o); // both made the identical change
    else if (eq(b, o)) out.push(...t); // only theirs effectively changed
    else if (eq(b, t)) out.push(...o); // only ours effectively changed
    else {
      conflicts.push({ baseLines: b, oursLines: o, theirsLines: t });
      out.push('<<<<<<< ours', ...o, '=======', ...t, '>>>>>>> theirs');
    }
    basePos = hi;
    i = j;
  }
  out.push(...baseLines.slice(basePos));
  return { clean: conflicts.length === 0, merged: out.join('\n'), conflicts };
}

// ── ancestry ─────────────────────────────────────────────────────────────────

// Nearest revision reachable from both heads through parentRevisionIds (BFS,
// so the first meeting point is the closest); null when histories share nothing.
export function commonAncestor(state: WorkspaceGraph, revA: RevisionId, revB: RevisionId): RevisionId | null {
  for (const id of [revA, revB]) {
    if (!state.revisions.has(id)) throw new Error(`commonAncestor: unknown revision ${id}`);
  }
  const parents = (id: RevisionId) => state.revisions.get(id)?.parentRevisionIds ?? [];
  const ancestorsOfA = new Set<RevisionId>();
  for (const q = [revA]; q.length; ) {
    const id = q.shift()!;
    if (ancestorsOfA.has(id)) continue;
    ancestorsOfA.add(id);
    q.push(...parents(id));
  }
  const seen = new Set<RevisionId>();
  for (const q = [revB]; q.length; ) {
    const id = q.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    if (ancestorsOfA.has(id)) return id;
    q.push(...parents(id));
  }
  return null;
}

// ── the merge proposal ───────────────────────────────────────────────────────

// Compute diff3 between two heads of one chunk and raise it as a proposal.
// Unrelated histories merge against an empty base: everything conflicts, both
// sides survive. Accepting yields one revision with both heads as parents.
export function proposeMerge(
  ctx: TxCtx,
  opts: { chunkId: ChunkId; oursRevisionId: RevisionId; theirsRevisionId: RevisionId },
): { proposalId: ProposalId; clean: boolean } {
  const { chunkId, oursRevisionId: ours, theirsRevisionId: theirs } = opts;
  for (const id of [ours, theirs]) {
    const rev = ctx.state.revisions.get(id);
    if (!rev) throw new Error(`merge: unknown revision ${id}`);
    if (rev.chunkId !== chunkId) throw new Error(`merge: revision ${id} belongs to ${rev.chunkId}, not ${chunkId}`);
  }
  const ancestor = commonAncestor(ctx.state, ours, theirs);
  const base = ancestor ? revisionText(ctx.state, ancestor) : '';
  const result = diff3(base, revisionText(ctx.state, ours), revisionText(ctx.state, theirs));
  const n = result.conflicts.length;
  const { proposalId } = propose(ctx, {
    kind: 'merge',
    basisRevisionIds: [ours, theirs],
    targetChunkIds: [chunkId],
    payload: [{ op: 'revise', chunkId, text: result.merged, mergeParentRevisionIds: [ours, theirs] }],
    note: result.clean ? 'clean merge' : `merge with ${n} conflict${n === 1 ? '' : 's'}`,
  });
  return { proposalId, clean: result.clean };
}
