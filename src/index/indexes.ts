// Derived indexes: computed facts over kernel truth, disposable and rebuildable.
// Nothing here determines identity; an index entry is keyed by revision, and a
// query answers only from currently visible, non-redacted evidence — so query
// helpers re-check state, which lets deletion and redaction reflow through even
// a stale index without a rebuild.

import { METHOD_SENTENCES, METHOD_WORDS, decomposeText } from '../kernel/decompose';
import { currentRevision, revisionText, type WorkspaceGraph } from '../kernel/state';
import type { BlobHash, ChunkId, Revision, RevisionId } from '../kernel/types';
import { isCompositeMediaType } from '../kernel/types';

export interface TermPosting {
  chunkId: ChunkId;
  revisionId: RevisionId;
  start: number; // UTF-16 code-unit offset into the revision's text
}

export interface EchoSpan {
  chunkId: ChunkId;
  revisionId: RevisionId;
  start: number;
  end: number;
}

export interface Indexes {
  term: Map<string, TermPosting[]>; // word/icu@1 token (NFC, lowercased) -> postings
  interning: Map<BlobHash, RevisionId[]>; // exact-content equality, history included
  echo: Map<string, EchoSpan[]>; // normalized sentence key -> spans, current revisions only
}

const normalizeToken = (s: string) => s.normalize('NFC').toLowerCase();
const echoKey = (s: string) => s.normalize('NFC').toLowerCase().replace(/\s+/g, ' ').trim();

const push = <K, V>(map: Map<K, V[]>, key: K, value: V) => {
  const arr = map.get(key);
  if (arr) arr.push(value);
  else map.set(key, [value]);
};

const wordSpans = (text: string) => decomposeText(text, METHOD_WORDS);

// Is this index entry still answerable evidence? Current revision of a live
// chunk, not redacted. One staleness definition everywhere.
function visible(state: WorkspaceGraph, chunkId: ChunkId, revisionId: RevisionId): boolean {
  const chunk = state.chunks.get(chunkId);
  if (!chunk || chunk.tombstoned || chunk.currentRevisionId !== revisionId) return false;
  return !state.revisions.get(revisionId)?.redacted;
}

// term and echo index current revisions of non-tombstoned chunks; interning
// spans ALL non-redacted revisions (history included) for exact-content
// equality. Composites are skipped everywhere: their blobs are join config,
// not content.
export function buildIndexes(state: WorkspaceGraph): Indexes {
  const term = new Map<string, TermPosting[]>();
  const interning = new Map<BlobHash, RevisionId[]>();
  const echo = new Map<string, EchoSpan[]>();

  for (const rev of state.revisions.values()) {
    if (rev.redacted || isCompositeMediaType(rev.mediaType)) continue;
    push(interning, rev.blobHash, rev.id);
  }

  for (const chunk of state.chunks.values()) {
    if (chunk.tombstoned) continue;
    const rev = state.revisions.get(chunk.currentRevisionId);
    if (!rev || rev.redacted || isCompositeMediaType(rev.mediaType)) continue;
    const text = state.blobs.get(rev.blobHash)?.text ?? '';
    for (const w of wordSpans(text)) {
      push(term, normalizeToken(text.slice(w.start, w.end)), { chunkId: chunk.id, revisionId: rev.id, start: w.start });
    }
    for (const s of decomposeText(text, METHOD_SENTENCES)) {
      const sentence = text.slice(s.start, s.end);
      if (wordSpans(sentence).length < 3) continue; // short sentences echo everywhere; not evidence
      push(echo, echoKey(sentence), { chunkId: chunk.id, revisionId: rev.id, start: s.start, end: s.end });
    }
  }
  return { term, interning, echo };
}

// Every query token must prefix-match some indexed token of the chunk; rank by
// total matched postings, chunk id as deterministic tiebreak.
export function searchChunks(state: WorkspaceGraph, ix: Indexes, query: string): ChunkId[] {
  const qtokens = wordSpans(query).map((w) => normalizeToken(query.slice(w.start, w.end)));
  if (!qtokens.length) return [];
  const perToken = qtokens.map(() => new Map<ChunkId, number>());
  for (const [token, postings] of ix.term) {
    for (const [i, qt] of qtokens.entries()) {
      if (!token.startsWith(qt)) continue;
      const m = perToken[i];
      for (const p of postings) {
        if (visible(state, p.chunkId, p.revisionId)) m.set(p.chunkId, (m.get(p.chunkId) ?? 0) + 1);
      }
    }
  }
  const [first, ...rest] = perToken;
  const scores: [ChunkId, number][] = [];
  for (const [chunkId, n] of first) {
    if (rest.every((m) => m.has(chunkId))) {
      scores.push([chunkId, n + rest.reduce((sum, m) => sum + m.get(chunkId)!, 0)]);
    }
  }
  return scores.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(([id]) => id);
}

// Sentences of this chunk that also appear, verbatim after normalization, in
// other visible chunks. Soft findings: promotion is someone else's explicit act.
export function echoesOf(
  state: WorkspaceGraph,
  ix: Indexes,
  chunkId: ChunkId,
): { key: string; text: string; others: ChunkId[] }[] {
  const out: { key: string; text: string; others: ChunkId[] }[] = [];
  for (const [key, spans] of ix.echo) {
    const mine = spans.find((s) => s.chunkId === chunkId && visible(state, s.chunkId, s.revisionId));
    if (!mine) continue;
    const others = [
      ...new Set(
        spans
          .filter((s) => s.chunkId !== chunkId && visible(state, s.chunkId, s.revisionId))
          .map((s) => s.chunkId),
      ),
    ];
    if (!others.length) continue;
    out.push({ key, text: revisionText(state, mine.revisionId).slice(mine.start, mine.end), others });
  }
  return out;
}

// Chunks whose CURRENT revision carries this exact content.
export function duplicatesOf(state: WorkspaceGraph, ix: Indexes, blobHash: BlobHash): ChunkId[] {
  const out = new Set<ChunkId>();
  for (const revId of ix.interning.get(blobHash) ?? []) {
    const rev = state.revisions.get(revId);
    if (rev && visible(state, rev.chunkId, revId)) out.add(rev.chunkId);
  }
  return [...out];
}

// Earliest visible bearer of this content — a query result, never a stored
// crown: redact or tombstone the earliest and attribution reflows to the next
// eligible revision (deletion.md).
export function firstSeen(state: WorkspaceGraph, ix: Indexes, blobHash: BlobHash): Revision | null {
  let best: Revision | null = null;
  for (const revId of ix.interning.get(blobHash) ?? []) {
    const rev = state.revisions.get(revId);
    if (!rev || rev.redacted) continue;
    const chunk = state.chunks.get(rev.chunkId);
    if (!chunk || chunk.tombstoned) continue;
    if (!best || rev.createdAt < best.createdAt || (rev.createdAt === best.createdAt && rev.id < best.id)) best = rev;
  }
  return best;
}

export type ProvenanceKind = 'human' | 'agent' | 'adapter' | 'external';

export function provenanceKind(state: WorkspaceGraph, chunkId: ChunkId): ProvenanceKind {
  const prefix = currentRevision(state, chunkId).createdBy.split(':', 1)[0];
  return prefix === 'human' || prefix === 'agent' || prefix === 'adapter' ? prefix : 'external';
}
