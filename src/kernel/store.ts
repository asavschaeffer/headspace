import type { Chunk } from './chunk';

// STUB: persistent, content-addressed store with history — in-memory only for the MVP.
export type Store = Map<string, Chunk>;

export interface Snapshot {
  root: string;
  chunks: Chunk[];
}

export function loadStore(snap: Snapshot): Store {
  return new Map(snap.chunks.map((c) => [c.id, c]));
}

// Hard chains: parent, siblings, own children.
export function neighbors(store: Store, id: string): Chunk[] {
  const c = store.get(id);
  if (!c) return [];
  const out: Chunk[] = [];
  const parent = c.parent ? store.get(c.parent) : undefined;
  if (parent) out.push(parent, ...parent.children.flatMap((s) => (s === id ? [] : store.get(s) ?? [])));
  out.push(...c.children.flatMap((k) => store.get(k) ?? []));
  return out;
}
