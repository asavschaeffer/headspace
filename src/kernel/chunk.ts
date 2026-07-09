// The one common object. Everything meaningful is a chunk.
export type ChunkKind = 'dir' | 'doc' | 'block';

export interface Chunk {
  id: string; // stable identity
  kind: ChunkKind;
  text: string; // versioned content
  children: string[]; // mutable arrangement — reordering does not touch the version
  version: number;
  provenance: { author: string; at: string }; // recorded authorship
  binding?: { path: string }; // link back to the authoritative filesystem object
  parent?: string;
}

const now = () => new Date().toISOString();

// STUB: only the current version is kept; a full version-history log is out of scope.
export function updateText(c: Chunk, text: string, author: string): Chunk {
  return { ...c, text, version: c.version + 1, provenance: { author, at: now() } };
}

export function fork(c: Chunk, id: string, author: string): Chunk {
  return { ...c, id, binding: undefined, version: 1, provenance: { author, at: now() } };
}

// Arrangement is mutable independently of content, so no version bump.
export function reorder(c: Chunk, from: number, to: number): Chunk {
  const children = [...c.children];
  const [moved] = children.splice(from, 1);
  children.splice(to, 0, moved);
  return { ...c, children };
}
