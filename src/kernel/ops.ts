import type { Chunk } from './chunk';
import { search } from './search';
import { neighbors, type Store } from './store';

// Select: choose relevant material — the focused chunk, its hard-chain
// neighborhood, and any search hits.
export function select(store: Store, focusId: string, query?: string): Chunk[] {
  const focus = store.get(focusId);
  if (!focus) return [];
  const picked = new Map<string, Chunk>([[focus.id, focus]]);
  for (const id of focus.children) {
    const c = store.get(id);
    if (c) picked.set(c.id, c);
  }
  for (const n of neighbors(store, focusId)) picked.set(n.id, n);
  if (query) for (const id of search(store, query)) picked.set(id, store.get(id)!);
  return [...picked.values()];
}

// Reduce: compile the selection into one bounded representation.
export function reduce(chunks: Chunk[], budget = 2000): string {
  let out = '';
  for (const c of chunks) {
    if (c.kind === 'dir') continue;
    if (out.length + c.text.length > budget) break;
    out += c.text + '\n\n';
  }
  return out.trimEnd();
}

// Generate: transform the reduced context through a model, returning a new chunk.
// STUB: the real agent dispatch (Anthropic API call) goes behind this exact signature.
export function generate(context: string, instruction: string): Chunk {
  const text =
    `[stub proposal] instruction: "${instruction}"\n\n` +
    `context seen (${context.length} chars):\n${context.slice(0, 240)}${context.length > 240 ? '…' : ''}`;
  return {
    id: `proposal-${crypto.randomUUID().slice(0, 8)}`,
    kind: 'block',
    text,
    children: [],
    version: 1,
    provenance: { author: 'agent:stub', at: new Date().toISOString() },
  };
}
