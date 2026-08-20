// Decomposition exposes smaller parts of one immutable revision on demand.
// Parts are addresses, not chunks: nothing here mints durable identity.
// Methods are versioned because segmentation is an interpretation.

import { revisionText, type SubstrateState } from './state';
import type { RevisionId, SpanAddress } from './types';
import { MEDIA_COMPOSITE } from './types';

export const METHOD_BLOCKS = 'md/blocks@1';
export const METHOD_SENTENCES = 'sent/icu@1';
export const METHOD_WORDS = 'word/icu@1';
export const METHOD_RAW = 'raw@1'; // a caller-chosen span with no segmentation claim

export interface DerivedPart {
  address: SpanAddress;
  text: string;
  kind: 'block' | 'sentence' | 'word';
}

interface Span {
  start: number;
  end: number;
}

// Markdown block segmentation: paragraphs and list runs split on blank lines,
// ATX headings stand alone, fenced code blocks stay whole. Offsets index the
// source text exactly (UTF-16 code units), so spans round-trip.
export function mdBlockSpans(text: string): Span[] {
  const spans: Span[] = [];
  let start = -1;
  let end = -1;
  let inFence = false;
  let fenceChar = '';
  let offset = 0;
  const flush = () => {
    if (start >= 0 && end > start) spans.push({ start, end });
    start = -1;
    end = -1;
  };
  for (const rawLine of text.split('\n')) {
    const lineStart = offset;
    offset += rawLine.length + 1;
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    const trimmed = line.trim();
    // Spans open at the line start (leading indentation is content — list
    // continuations, indented code) and close after the last non-whitespace.
    const lineContentEnd = lineStart + line.replace(/\s+$/, '').length;
    const fenceMatch = /^(`{3,}|~{3,})/.exec(trimmed);

    if (inFence) {
      if (trimmed !== '') end = lineContentEnd;
      if (fenceMatch && fenceMatch[1][0] === fenceChar) {
        inFence = false;
        flush();
      }
      continue;
    }
    if (fenceMatch) {
      flush();
      start = lineStart;
      end = lineContentEnd;
      inFence = true;
      fenceChar = fenceMatch[1][0];
      continue;
    }
    if (trimmed === '') {
      flush();
      continue;
    }
    if (/^#{1,6}\s/.test(trimmed)) {
      flush();
      spans.push({ start: lineStart, end: lineContentEnd });
      continue;
    }
    if (start < 0) start = lineStart;
    end = lineContentEnd;
  }
  flush();
  return spans;
}

type Granularity = 'sentence' | 'word';

function segmentSpans(text: string, granularity: Granularity): Span[] {
  const out: Span[] = [];
  const Segmenter = (Intl as unknown as { Segmenter?: new (loc: undefined, o: { granularity: Granularity }) => {
    segment(t: string): Iterable<{ segment: string; index: number; isWordLike?: boolean }>;
  } }).Segmenter;
  if (Segmenter) {
    const seg = new Segmenter(undefined, { granularity });
    for (const s of seg.segment(text)) {
      if (granularity === 'word' && !s.isWordLike) continue;
      const lead = s.segment.length - s.segment.trimStart().length;
      const len = s.segment.trim().length;
      if (len > 0) out.push({ start: s.index + lead, end: s.index + lead + len });
    }
    return out;
  }
  // Fallback where Intl.Segmenter is unavailable.
  const re = granularity === 'sentence' ? /[^.!?\n]+[.!?]*/g : /[\p{L}\p{N}]+/gu;
  for (const m of text.matchAll(re)) {
    const raw = m[0];
    const lead = raw.length - raw.trimStart().length;
    const len = raw.trim().length;
    if (len > 0) out.push({ start: (m.index ?? 0) + lead, end: (m.index ?? 0) + lead + len });
  }
  return out;
}

export function decomposeText(text: string, method: string): Array<Span & { kind: DerivedPart['kind'] }> {
  switch (method) {
    case METHOD_BLOCKS:
      return mdBlockSpans(text).map((s) => ({ ...s, kind: 'block' as const }));
    case METHOD_SENTENCES:
      return segmentSpans(text, 'sentence').map((s) => ({ ...s, kind: 'sentence' as const }));
    case METHOD_WORDS:
      return segmentSpans(text, 'word').map((s) => ({ ...s, kind: 'word' as const }));
    default:
      throw new Error(`unknown decomposition method: ${method}`);
  }
}

export function decomposeRevision(state: SubstrateState, revisionId: RevisionId, method: string): DerivedPart[] {
  const rev = state.revisions.get(revisionId);
  if (!rev) throw new Error(`decompose: unknown revision ${revisionId}`);
  if (rev.mediaType === MEDIA_COMPOSITE) {
    throw new Error('decompose: composites decompose through their children, not their blob');
  }
  const text = revisionText(state, revisionId);
  return decomposeText(text, method).map(({ start, end, kind }) => ({
    address: { revisionId, method, start, end },
    text: text.slice(start, end),
    kind,
  }));
}
