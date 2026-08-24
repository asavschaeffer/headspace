// String fractional indexing for occurrence positions. Sibling order is the
// lexicographic order of keys, so a reorder never renumbers other siblings and
// precision never runs out (keys grow by a character instead).
//
// Keys are non-empty strings over an alphabet in ASCII order, and never end in
// the minimum digit, which guarantees a key exists strictly between any two.

const DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const BASE = DIGITS.length;

// a < m < b, where '' stands for the open end on either side.
function midpoint(a: string, b: string): string {
  if (b !== '') {
    let n = 0;
    while (a[n] !== undefined && a[n] === b[n]) n++;
    if (n > 0) return b.slice(0, n) + midpoint(a.slice(n), b.slice(n));
  }
  const da = a === '' ? 0 : DIGITS.indexOf(a[0]);
  const db = b === '' ? BASE : DIGITS.indexOf(b[0]);
  if (db - da > 1) return DIGITS[Math.round((da + db) / 2)];
  // Adjacent leading digits: descend.
  if (a.length > 1) return a[0] + midpoint(a.slice(1), '');
  return DIGITS[da] + midpoint('', b.slice(1));
}

export function keyBetween(a: string | null, b: string | null): string {
  const lo = a ?? '';
  const hi = b ?? '';
  if (lo !== '' && hi !== '' && lo >= hi) throw new Error(`keyBetween: "${lo}" >= "${hi}"`);
  if (lo !== '' && lo.endsWith(DIGITS[0])) throw new Error(`keyBetween: key ends in minimum digit: "${lo}"`);
  return midpoint(lo, hi);
}

// n keys evenly spread between the bounds, by balanced bisection.
export function keysBetween(a: string | null, b: string | null, n: number): string[] {
  if (n <= 0) return [];
  const mid = keyBetween(a, b);
  const left = keysBetween(a, mid, Math.floor((n - 1) / 2));
  const right = keysBetween(mid, b, Math.ceil((n - 1) / 2));
  return [...left, mid, ...right];
}
