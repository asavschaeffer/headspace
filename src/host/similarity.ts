// Normalized Levenshtein similarity for driver block matching, pass 2 of the
// reconcile match (wiki/drivers.md): 1 means identical, 0 means disjoint.
//
// The DP is O(n*m) in time, so cost must be bounded before it is paid. Two
// guards do that, and both are stated in terms of the driver's 0.5 threshold:
// a cheap length test rejects pairs that provably cannot reach it, and a hard
// cap keeps any single comparison bounded no matter how large the blocks are.
// Reconcile runs this over (free manifest entries x file blocks), and the dev
// server serializes every request behind one workspace, so an unbounded pair
// would block the whole host, not just this call.

// Widest exact comparison, per side. 4096^2 cells is tens of milliseconds;
// beyond it the score comes from a bounded sample instead.
const MAX_EXACT_CHARS = 4096;
// Head and tail window kept from each side when the cap is exceeded.
const SAMPLE_CHARS = 1024;

export interface SimilarityAssessment {
  score: number;
  method: 'empty' | 'length-bound' | 'exact-levenshtein' | 'sampled-levenshtein';
  approximate: boolean;
}

export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev: number[] = Array.from({ length: b.length + 1 }, (_, j) => j);
  let cur: number[] = new Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= b.length; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[b.length];
}

// The ends of a long block are what an edit leaves recognizable, so the sample
// keeps both and drops the middle. Shorter than the window on either side means
// the whole string is kept, which makes this exact for anything under the cap.
const sample = (s: string): string =>
  s.length <= SAMPLE_CHARS * 2 ? s : s.slice(0, SAMPLE_CHARS) + s.slice(-SAMPLE_CHARS);

export function assessSimilarity(a: string, b: string): SimilarityAssessment {
  const max = Math.max(a.length, b.length);
  if (max === 0) return { score: 1, method: 'empty', approximate: false };
  // Levenshtein distance is at least the length difference, so a pair whose
  // lengths differ by more than half the longer one cannot score 0.5. Zero is
  // a floor here, not a measurement — the only consumer is that threshold.
  if (Math.abs(a.length - b.length) > max / 2) {
    return { score: 0, method: 'length-bound', approximate: false };
  }
  if (max <= MAX_EXACT_CHARS) {
    return { score: 1 - levenshtein(a, b) / max, method: 'exact-levenshtein', approximate: false };
  }
  // Above the cap the score is an approximation over head and tail. Blocks this
  // large are rare, and the alternative — refusing to match — would mint a new
  // chunk and propose a sever every time a big block is edited outside.
  const sa = sample(a);
  const sb = sample(b);
  return {
    score: 1 - levenshtein(sa, sb) / Math.max(sa.length, sb.length),
    method: 'sampled-levenshtein',
    approximate: true,
  };
}

export function similarity(a: string, b: string): number {
  return assessSimilarity(a, b).score;
}
