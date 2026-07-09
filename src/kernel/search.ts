import type { Store } from './store';

// STUB: embedding index + HDBSCAN clustering for soft chains — naive text match for now.
export function search(store: Store, query: string): Set<string> {
  const q = query.trim().toLowerCase();
  const hits = new Set<string>();
  if (!q) return hits;
  for (const c of store.values()) {
    if (c.text.toLowerCase().includes(q)) {
      hits.add(c.id);
      if (c.parent) hits.add(c.parent); // illuminate the containing star too
    }
  }
  return hits;
}
